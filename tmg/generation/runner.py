"""Pool generation: chunk plan, parallel chunk processes, resume, provenance. Generator-independent."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
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
from tmg.core.records import RECORDS_DIR
from tmg.core.workers import resolve_workers
from tmg.generation.base import ChunkSpec
from tmg.generation.generators import make_generator


class PoolError(RuntimeError):
    """A pool cannot run or did not complete (missing sources, changed chunks, failed chunks)."""


@dataclass
class LaunchedChunk:
    """What :func:`launch_chunk` knows once the chunk process returned."""
    spec: ChunkSpec
    exit_code: int
    elapsed_s: float            # wall time of the chunk process alone (preparation excluded)


def _free_gib(path: Path) -> float:
    """Free disk space (GiB) of the filesystem holding ``path`` (or its nearest existing parent)."""
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


# chunk inspection -----------------------------------------------------------------------------------
def _chunk_state(folder: Path, spec: ChunkSpec) -> tuple[str, list[str]]:
    """absent | partial | done | changed, with reasons."""
    if not folder.exists():
        return "absent", []
    if not chunk_is_done(folder):
        return "partial", []
    old_path = folder / CHUNK_SPEC
    if not old_path.exists():
        return "changed", [f"finished chunk has no {CHUNK_SPEC}"]
    old, new = ChunkSpec(**read_json(old_path)).identity(), spec.identity()
    if old != new:
        diffs = [k for k in new if old.get(k) != new[k]]
        return "changed", [f"finished chunk was made under another configuration: {diffs}"]
    return "done", []


def _relocate_spec(spec: ChunkSpec, scratch: Path, folder: Path) -> ChunkSpec:
    """Rewrite the paths of a spec prepared in ``scratch`` as if it had been prepared in ``folder``."""
    spec.folder = str(folder)
    spec.records_dir = str(folder / RECORDS_DIR)
    spec.generator_config = str(folder / Path(spec.generator_config).name)
    spec.expected_outputs = [str(folder / Path(p).relative_to(scratch)) for p in spec.expected_outputs]
    return spec


def _spec_of_finished_chunk(generator, chunk: Chunk, folder: Path) -> ChunkSpec:
    """The spec the current configuration would give ``chunk``, built in a scratch folder so that the finished
    chunk's files are not touched, with its paths pointing at ``folder`` for the identity comparison."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / chunk.name
        return _relocate_spec(generator.prepare_chunk(chunk, scratch), scratch, folder)


def _inspect_chunks(generator, chunks: list[Chunk], output: Path, own: set[int] | None = None) -> tuple[dict[int, str], list[str]]:
    """State of every chunk before anything runs (see :func:`_chunk_state`).

    Args:
        own: indices of the chunks this runner may run (a shard); None for all. Unfinished chunks of other shards are
            only reported (absent or partial), since another runner may be writing them.

    Returns:
        ``({chunk index: state}, [reasons of the changed chunks])``. Unfinished chunks of this runner are prepared in
        place, which writes the generator's config into their folder.
    """
    states: dict[int, str] = {}
    changed: list[str] = []
    for chunk in chunks:
        folder = chunk.folder(output)
        if chunk_is_done(folder):
            spec = _spec_of_finished_chunk(generator, chunk, folder)
        elif own is not None and chunk.index not in own:
            states[chunk.index] = "partial" if folder.exists() else "absent"
            continue
        else:
            spec = generator.prepare_chunk(chunk, folder)
        state, reasons = _chunk_state(folder, spec)
        states[chunk.index] = state
        if state == "changed":
            changed.append(f"{chunk.name}: {'; '.join(reasons)}")
    return states, changed


# chunk processes -------------------------------------------------------------------------------------
def _chunk_env(cluster: Mapping) -> dict[str, str]:
    """Environment of a chunk process: the caller's environment with the BLAS/OpenMP thread caps of
    ``cluster.threads_per_worker`` and unbuffered Python output (so logs show progress while a chunk runs)."""
    threads = str(int(cluster.get("threads_per_worker", 1)))
    return dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                PYTHONUNBUFFERED="1")


def launch_chunk(generator, chunk: Chunk, folder: Path, *, env: Mapping[str, str], log: Path,
                 extra: Mapping | None = None) -> LaunchedChunk:
    """Prepare ``chunk`` in ``folder``, write its spec and run the chunk process to completion.

    Args:
        generator: the generator adapter.
        chunk: the chunk to run.
        folder: the chunk folder; the caller decides what to do with an existing one (the runner clears a partial
            chunk, the baseline check removes the folder entirely).
        env: environment of the chunk process.
        log: file that receives the process's stdout and stderr.
        extra: entries merged into ``spec.extra`` before the spec is written (``disable_graft`` of the baseline check).

    Returns:
        The spec as written, the exit code and the wall time of the process.
    """
    spec = generator.prepare_chunk(chunk, folder)
    if extra:
        spec.extra.update(extra)
    spec_path = write_json(folder / CHUNK_SPEC, asdict(spec))
    cmd = generator.chunk_command(spec, spec_path)
    start = time.time()
    with open(log, "w") as lf:
        code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                              cwd=str(folder)).returncode
    return LaunchedChunk(spec=spec, exit_code=code, elapsed_s=time.time() - start)


def _run_chunk(generator, chunk: Chunk, output: Path, env: Mapping[str, str]) -> dict:
    """Run one pending chunk of the pool and return its result line (``CHUNK_DONE.json`` content on success)."""
    folder = chunk.folder(output)
    clear_partial_chunk(folder)
    log = output / LOGS_DIR / f"{chunk.name}.log"
    launched = launch_chunk(generator, chunk, folder, env=env, log=log)
    result = generator.chunk_result(launched.spec)
    if result is None or "error" in result:
        return {"chunk": chunk.name, "ok": False, "exit_code": launched.exit_code, "log": str(log),
                "elapsed_s": launched.elapsed_s, "error": None if result is None else result["error"]}
    result.update(chunk=chunk.name, ok=True, exit_code=launched.exit_code, elapsed_s=launched.elapsed_s, log=str(log))
    write_json(folder / CHUNK_DONE, result)
    return result


def _run_pending(generator, pending: list[Chunk], output: Path, env: Mapping[str, str], workers: int) -> list[dict]:
    """Run the pending chunks ``workers`` at a time, printing one line per finished chunk."""
    results = []
    if not pending:
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda chunk: _run_chunk(generator, chunk, output, env), pending):
            flag = "ok" if result["ok"] else "FAILED"
            print(f"[tmg] chunk {result['chunk']}: {flag} ({result.get('successes', '?')}/"
                  f"{result.get('attempts', '?')} successes, {result['elapsed_s']:.0f}s)", flush=True)
            results.append(result)
    return results


# summary --------------------------------------------------------------------------------------------
def _summarize(plain: Mapping, generator, chunks: list[Chunk], output: Path) -> dict:
    """Collect every chunk's ``CHUNK_DONE.json`` into the pool summary, write ``DONE.json`` and print one line."""
    generation = plain["generation"]
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
    # generation.shard [i, n]: this runner runs the chunks whose index is i modulo n, so n runners (e.g. one container per
    # GPU) share one pool; the runner that finds every chunk done writes DONE.json
    shard = generation.get("shard")
    own = None
    if shard:
        i, n = (int(v) for v in shard)
        if not 0 <= i < n:
            raise PoolError(f"generation.shard {shard}: expected [index, count] with 0 <= index < count")
        own = {c.index for c in chunks if c.index % n == i}

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

    states, changed = _inspect_chunks(generator, chunks, output, own)  # inspect every chunk before anything runs
    if changed:
        if str(generation.get("on_config_change", "error")) != "rebuild":
            raise PoolError("finished chunks differ from the requested configuration (set generation.on_config_change="
                            "rebuild to regenerate them):\n  " + "\n  ".join(changed))
        for chunk in chunks:
            if states[chunk.index] == "changed" and (own is None or chunk.index in own):
                shutil.rmtree(chunk.folder(output))
                states[chunk.index] = "absent"

    dump_yaml(cfg, output / RESOLVED_CONFIG)
    write_json(output / POOL_PLAN, {"chunks": plan_as_dict(chunks), "generator": generator.name})
    write_provenance(output, {"command": "generate", "output": str(output)})

    pending = [c for c in chunks if states[c.index] != "done" and (own is None or c.index in own)]
    workers, explained = resolve_workers(cluster.get("workers", "auto"), int(cluster.get("max_workers", 16)))
    done = sum(1 for c in chunks if states[c.index] == "done")
    mine = f" (shard {shard[0]}/{shard[1]}: {len(own)} chunks)" if own is not None else ""
    print(f"[tmg] pool {output}: {len(chunks)} chunks, {done} done, {len(pending)} to run here{mine}; "
          f"workers {explained}", flush=True)
    results = _run_pending(generator, pending, output, _chunk_env(cluster), workers)
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise PoolError(f"{len(failed)} chunk(s) failed; see " + ", ".join(r["log"] for r in failed))
    if own is not None:
        waiting = [c.name for c in chunks if not chunk_is_done(c.folder(output))]
        if waiting:
            print(f"[tmg] pool shard {shard[0]}/{shard[1]} done; {len(waiting)} chunk(s) of other shards still running, "
                  f"DONE.json is written by the last runner", flush=True)
            return {"partial": True, "shard": list(shard), "waiting": waiting}
    return _summarize(plain, generator, chunks, output)
