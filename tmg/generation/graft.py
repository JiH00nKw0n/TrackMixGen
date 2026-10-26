"""Runtime grafting of callbacks into official functions (no copy of the official source in this repo).

The official function's source is read with ``inspect.getsource``, its sha256 compared with the pinned value,
parsed with ``ast``, and small statements are inserted or replaced at anchors identified by the names they
assign. The modified tree is compiled with the original function's globals plus the callback object and
swapped into the class. The official module file is never changed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable


class GraftError(RuntimeError):
    """An anchor was not found, a hash did not match, or the source could not be edited."""


def source_hash(func: Callable) -> str:
    """sha256 of ``inspect.getsource(func)`` (the value pinned in the generator config)."""
    return hashlib.sha256(inspect.getsource(func).encode()).hexdigest()


def _iter_bodies(node):
    """Yield every statement list (body, orelse, finalbody, handlers' bodies) under ``node``, recursively."""
    for field in ("body", "orelse", "finalbody"):
        stmts = getattr(node, field, None)
        if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
            yield stmts
            for child in stmts:
                yield from _iter_bodies(child)
    for handler in getattr(node, "handlers", []) or []:
        yield from _iter_bodies(handler)


def _assigns_name(stmt, name: str) -> bool:
    """True when ``stmt`` is an assignment whose target (or tuple target) includes ``name``."""
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return True
            if isinstance(target, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == name for e in target.elts):
                return True
    return False


class Graft:
    """Edit one official function's AST and compile a replacement.

    Typical use::

        g = Graft(Cls.method, expected_hash=..., label="Cls.method")
        g.insert_after_assign("x", "_cb.after_x(x)")
        Cls.method = g.compile({"_cb": callbacks})

    Every edit method returns ``self`` and records a human-readable description in ``edits``.
    """
    def __init__(self, func: Callable, *, expected_hash: str | None, label: str):
        """Read and hash the function source and parse it.

        Args:
            func: the official function (unbound method).
            expected_hash: pinned sha256 of its source, or None to skip the check.
            label: name used in error messages.

        Raises:
            GraftError: the installed source differs from ``expected_hash``.
        """
        self.func = func
        self.label = label
        source = inspect.getsource(func)
        found = hashlib.sha256(source.encode()).hexdigest()
        if expected_hash is not None and found != expected_hash:
            raise GraftError(f"{label}: installed source hash {found} differs from the pinned {expected_hash}; "
                             "the official package changed, re-check the graft anchors before running")
        self.hash = found
        self.start_line = inspect.getsourcelines(func)[1]
        self.tree = ast.parse(textwrap.dedent(source))
        self.node = self.tree.body[0]
        if not isinstance(self.node, ast.FunctionDef):
            raise GraftError(f"{label}: source does not start with a function definition")
        self.node.decorator_list = []
        self.edits: list[str] = []

    # locating -------------------------------------------------------------------------------------
    def _find_assign(self, name: str, occurrence: int = 0):
        """``(statement list, index)`` of the ``occurrence``-th assignment to ``name`` in document order."""
        hits = []
        for stmts in _iter_bodies(self.node):
            for i, stmt in enumerate(stmts):
                if _assigns_name(stmt, name):
                    hits.append((stmts, i))
        if len(hits) <= occurrence:
            raise GraftError(f"{self.label}: no assignment #{occurrence} to '{name}'")
        return hits[occurrence]

    def _find_for(self, target_name: str):
        """``(statement list, index)`` of the unique ``for <target_name> in ...`` loop."""
        hits = []
        for stmts in _iter_bodies(self.node):
            for i, stmt in enumerate(stmts):
                if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and stmt.target.id == target_name:
                    hits.append((stmts, i))
        if len(hits) != 1:
            raise GraftError(f"{self.label}: expected exactly one 'for {target_name} in ...', found {len(hits)}")
        return hits[0]

    @staticmethod
    def _parse_stmt(code: str) -> ast.stmt:
        """Parse exactly one statement from ``code``."""
        body = ast.parse(textwrap.dedent(code)).body
        if len(body) != 1:
            raise GraftError("exactly one statement expected")
        return body[0]

    # edits ----------------------------------------------------------------------------------------
    def insert_after_assign(self, name: str, code: str, occurrence: int = 0) -> Graft:
        """Insert statement ``code`` right after the ``occurrence``-th assignment to ``name``."""
        stmts, i = self._find_assign(name, occurrence)
        stmts.insert(i + 1, self._parse_stmt(code))
        self.edits.append(f"after '{name} = ...': {code.strip()}")
        return self

    def replace_assign_value(self, name: str, code: str, occurrence: int = 0) -> Graft:
        """Replace the right-hand side of the ``occurrence``-th assignment to ``name`` with expression ``code``."""
        stmts, i = self._find_assign(name, occurrence)
        value = ast.parse(textwrap.dedent(code), mode="eval").body
        stmts[i].value = value
        self.edits.append(f"'{name} = {code.strip()}'")
        return self

    def replace_for_iter(self, target_name: str, code: str) -> Graft:
        """Replace the iterable of ``for <target_name> in ...`` with expression ``code``."""
        stmts, i = self._find_for(target_name)
        stmts[i].iter = ast.parse(textwrap.dedent(code), mode="eval").body
        self.edits.append(f"'for {target_name} in {code.strip()}'")
        return self

    def insert_at_for_start(self, target_name: str, code: str) -> Graft:
        """Insert statement ``code`` as the first statement of the body of ``for <target_name> in ...``."""
        stmts, i = self._find_for(target_name)
        stmts[i].body.insert(0, self._parse_stmt(code))
        self.edits.append(f"first statement of 'for {target_name}': {code.strip()}")
        return self

    def insert_before_final_return(self, code: str) -> Graft:
        """Insert statement ``code`` just before the function's final ``return``."""
        body = self.node.body
        if not isinstance(body[-1], ast.Return):
            raise GraftError(f"{self.label}: function does not end with a return")
        body.insert(len(body) - 1, self._parse_stmt(code))
        self.edits.append(f"before final return: {code.strip()}")
        return self

    # compile --------------------------------------------------------------------------------------
    def compile(self, extra_globals: dict) -> Callable:
        """Compile the edited function with the original globals plus ``extra_globals``.

        Returns:
            The new function; its ``__tmg_graft__`` attribute lists the label, hash and edits.
        """
        module = ast.Module(body=[self.node], type_ignores=[])
        ast.fix_missing_locations(module)
        ast.increment_lineno(module, self.start_line - 1)
        filename = inspect.getsourcefile(self.func) or "<grafted>"
        code = compile(module, filename, "exec")
        namespace = dict(self.func.__globals__)
        namespace.update(extra_globals)
        exec(code, namespace)
        new = namespace[self.node.name]
        new.__qualname__ = getattr(self.func, "__qualname__", self.node.name)
        new.__tmg_graft__ = {"label": self.label, "hash": self.hash, "edits": list(self.edits)}
        return new
