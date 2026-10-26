"""Pool generation: chunk plan, parallel chunk processes, resume, provenance. Generator-independent."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import (
    CHUNK_DONE,
    CHUNK_SPEC,
    LOGS_DIR,
    POOL_DONE,
    POOL_PLAN,
    RESOLVED_CONFIG,
    Chunk,
    chunk_is_done,
    clear_partial_chunk,
    plan_as_dict,
    plan_chunks,
    read_json,
    write_json,
)
from tmg.core.provenance import write_provenance
from tmg.core.workers import resolve_workers
from tmg.generation.base import ChunkSpec
from tmg.generation.generators import make_generator


class PoolError(RuntimeError):
    """A pool cannot run or did not complete (missing sources, changed chunks, failed chunks)."""


def _free_gib(path: Path) -> float:
    """Free disk space (GiB) of the filesystem holding ``path`` (or its nearest existing parent)."""
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def _chunk_state(generator, chunk: Chunk, folder: Path, spec: ChunkSpec) -> tuple[str, list[str]]:
    """absent | partial | done | changed, with reasons."""
    if not folder.exists():
        return "absent", []
    if not chunk_is_done(folder):
        return "partial", []
    old_path = folder / CHUNK_SPEC
    if not old_path.exists():
        return "changed", [f"finished chunk has no {CHUNK_SPEC}"]
    old = ChunkSpec(**read_json(old_path))
    if old.identity() != spec.identity():
        diffs = [k for k in spec.identity() if old.identity().get(k) != spec.identity()[k]]
        return "changed", [f"finished chunk was made under another configuration: {diffs}"]
    return "done", []


def run_pool(cfg) -> dict:
    """Generate one pool: plan chunks, reuse finished ones, run the rest in parallel, write ``DONE.json``.

    Args:
        cfg: the composed ``generate`` config.

    Returns:
        The pool summary written to ``DONE.json`` (attempts, successes, per-chunk results, plan).

    Raises:
        PoolError: sources missing, too little disk, finished chunks made under another configuration (unless
            ``generation.on_config_change=rebuild``), or failed chunks.
    """
    plain = resolved(cfg)
    generator = make_generator(plain)
    output = Path(str(plain["output"])).expanduser()
    cluster = dict(plain["cluster"])
    generation = dict(plain["generation"])
    chunks = plan_chunks(int(generation["attempts"]), int(generation["chunk"]), int(generation["seed_offset"]))

    problems = generator.check_sources()
    if problems:
        raise PoolError("; ".join(problems))
    free = _free_gib(output)
    if free < float(cluster.get("min_free_disk_gib", 0)):
        raise PoolError(f"only {free:.1f} GiB free under {output}; cluster.min_free_disk_gib is {cluster['min_free_disk_gib']}")

    output.mkdir(parents=True, exist_ok=True)
    (output / LOGS_DIR).mkdir(exist_ok=True)
    if (output / POOL_DONE).exists():
        done = read_json(output / POOL_DONE)
        if done.get("plan") == plan_as_dict(chunks):
            print(f"[tmg] pool already complete: {output}", flush=True)
            return done

    # inspect every chunk before anything runs
    specs: dict[int, ChunkSpec] = {}
    states: dict[int, str] = {}
    changed: list[str] = []
    for chunk in chunks:
        folder = chunk.folder(output)
        spec = generator.prepare_chunk(chunk, folder) if not chunk_is_done(folder) else None
        if spec is None:
            # finished chunk: build the spec in a scratch folder to compare identities without touching it
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                spec = generator.prepare_chunk(chunk, Path(tmp) / chunk.name)
                spec.folder = str(folder)
                spec.records_dir = str(folder / "records")
                spec.generator_config = str(folder / Path(spec.generator_config).name)
                spec.expected_outputs = [str(folder / Path(p).relative_to(Path(tmp) / chunk.name)) for p in spec.expected_outputs]
        state, reasons = _chunk_state(generator, chunk, folder, spec)
        specs[chunk.index] = spec
        states[chunk.index] = state
        if state == "changed":
            changed.append(f"{chunk.name}: {'; '.join(reasons)}")
    if changed:
        if str(generation.get("on_config_change", "error")) != "rebuild":
            raise PoolError("finished chunks differ from the requested configuration (set generation.on_config_change="
                            "rebuild to regenerate them):\n  " + "\n  ".join(changed))
        for chunk in chunks:
            if states[chunk.index] == "changed":
                shutil.rmtree(chunk.folder(output))
                states[chunk.index] = "absent"

    dump_yaml(cfg, output / RESOLVED_CONFIG)
    write_json(output / POOL_PLAN, {"chunks": plan_as_dict(chunks), "generator": generator.name})
    write_provenance(output, {"command": "generate", "output": str(output)})

    pending = [c for c in chunks if states[c.index] != "done"]
    workers, explained = resolve_workers(cluster.get("workers", "auto"), int(cluster.get("max_workers", 16)))
    print(f"[tmg] pool {output}: {len(chunks)} chunks, {len(chunks) - len(pending)} done, {len(pending)} to run; "
          f"workers {explained}", flush=True)
    threads = str(int(cluster.get("threads_per_worker", 1)))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               PYTHONUNBUFFERED="1")

    def run_one(chunk: Chunk) -> dict:
        folder = chunk.folder(output)
        clear_partial_chunk(folder)
        spec = generator.prepare_chunk(chunk, folder)
        spec_path = write_json(folder / CHUNK_SPEC, asdict(spec))
        cmd = generator.chunk_command(spec, spec_path)
        log = output / LOGS_DIR / f"{chunk.name}.log"
        start = time.time()
        with open(log, "w") as lf:
            code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                  cwd=str(folder)).returncode
        result = generator.chunk_result(spec)
        elapsed = time.time() - start
        if result is None or "error" in result:
            return {"chunk": chunk.name, "ok": False, "exit_code": code, "log": str(log), "elapsed_s": elapsed,
                    "error": None if result is None else result["error"]}
        result.update(chunk=chunk.name, ok=True, exit_code=code, elapsed_s=elapsed, log=str(log))
        write_json(folder / CHUNK_DONE, result)
        return result

    results = []
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(run_one, pending):
                flag = "ok" if result["ok"] else "FAILED"
                print(f"[tmg] chunk {result['chunk']}: {flag} ({result.get('successes', '?')}/"
                      f"{result.get('attempts', '?')} successes, {result['elapsed_s']:.0f}s)", flush=True)
                results.append(result)
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise PoolError(f"{len(failed)} chunk(s) failed; see " + ", ".join(r["log"] for r in failed))

    done_results = [read_json(c.folder(output) / CHUNK_DONE) for c in chunks]
    summary = {
        "generator": generator.name, "task": plain["task"]["name"], "scene": plain["scene"]["name"],
        "method": plain["method"]["name"], "generation": generation["name"],
        "attempts": sum(int(r["attempts"]) for r in done_results),
        "successes": sum(int(r["successes"]) for r in done_results),
        "chunks": done_results, "plan": plan_as_dict(chunks), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary["success_rate"] = summary["successes"] / max(summary["attempts"], 1)
    write_json(output / POOL_DONE, summary)
    print(f"[tmg] pool complete: {summary['successes']}/{summary['attempts']} successes "
          f"({100 * summary['success_rate']:.1f}%) -> {output}", flush=True)
    return summary
