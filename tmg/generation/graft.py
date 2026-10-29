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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field


class GraftError(RuntimeError):
    """An anchor was not found, a hash did not match, or the source could not be edited."""


def text_hash(source: str) -> str:
    """sha256 hex digest of a function's source text."""
    return hashlib.sha256(source.encode()).hexdigest()


def source_hash(func: Callable) -> str:
    """sha256 of ``inspect.getsource(func)`` (the value pinned in the generator config)."""
    return text_hash(inspect.getsource(func))


def _iter_bodies(node):
    """Yield every statement list (body, orelse, finalbody, handlers' bodies) under ``node``, recursively."""
    for attribute in ("body", "orelse", "finalbody"):
        stmts = getattr(node, attribute, None)
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
        found = text_hash(source)
        if expected_hash is not None and found != expected_hash:
            raise GraftError(f"{label}: installed source hash {found} differs from the pinned {expected_hash}; "
                             "the official package changed, re-check the graft anchors before running")
        self.hash = found
        self.start_line = inspect.getsourcelines(func)[1]
        self.tree = ast.parse(textwrap.dedent(source))
        self.node = self.tree.body[0]
        if not isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef)):   # Isaac Lab's generate/execute are async
            raise GraftError(f"{label}: source does not start with a function definition")
        self.node.decorator_list = []
        self.edits: list[str] = []

    # locating -------------------------------------------------------------------------------------
    def _statements(self, predicate: Callable[[ast.stmt], bool]) -> Iterator[tuple[list, int]]:
        """``(statement list, index)`` of every statement of the function for which ``predicate`` holds, in the
        order ``_iter_bodies`` visits them (outer bodies before nested ones)."""
        for stmts in _iter_bodies(self.node):
            for i, stmt in enumerate(stmts):
                if predicate(stmt):
                    yield stmts, i

    def _find_assign(self, name: str, occurrence: int = 0):
        """``(statement list, index)`` of the ``occurrence``-th assignment to ``name`` in document order."""
        hits = list(self._statements(lambda stmt: _assigns_name(stmt, name)))
        if len(hits) <= occurrence:
            raise GraftError(f"{self.label}: no assignment #{occurrence} to '{name}'")
        return hits[occurrence]

    def _find_for(self, target_name: str):
        """``(statement list, index)`` of the unique ``for <target_name> in ...`` loop."""
        hits = list(self._statements(
            lambda stmt: isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and stmt.target.id == target_name))
        if len(hits) != 1:
            raise GraftError(f"{self.label}: expected exactly one 'for {target_name} in ...', found {len(hits)}")
        return hits[0]

    def _find_returns(self):
        """``[(statement list, index)]`` of every ``return`` under the function in document order (source line, then
        column), nested function bodies included since ``_iter_bodies`` visits them too."""
        hits = list(self._statements(lambda stmt: isinstance(stmt, ast.Return)))
        return sorted(hits, key=lambda h: (getattr(h[0][h[1]], "lineno", 0), getattr(h[0][h[1]], "col_offset", 0)))

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

    def insert_before_return(self, code: str, occurrence: int, expected_returns: int | None = None) -> Graft:
        """Insert statement ``code`` before the ``occurrence``-th ``return`` (document order).

        Args:
            expected_returns: when given, the function must have exactly this many returns (guards the anchor).
        """
        hits = self._find_returns()
        if expected_returns is not None and len(hits) != expected_returns:
            raise GraftError(f"{self.label}: expected {expected_returns} return statements, found {len(hits)}")
        if occurrence >= len(hits):
            raise GraftError(f"{self.label}: no return #{occurrence} (found {len(hits)})")
        stmts, i = hits[occurrence]
        stmts.insert(i, self._parse_stmt(code))
        self.edits.append(f"before return #{occurrence}: {code.strip()}")
        return self

    def name_final_return(self, name: str) -> Graft:
        """Rewrite the final ``return <expr>`` as ``<name> = <expr>; return <name>`` so the value gets an anchor."""
        body = self.node.body
        if not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise GraftError(f"{self.label}: function does not end with a valued return")
        original = body[-1]
        value = original.value
        assign = ast.copy_location(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value, type_comment=None), original)
        body[-1] = ast.copy_location(ast.Return(value=ast.Name(id=name, ctx=ast.Load())), original)
        body.insert(len(body) - 1, assign)
        self.edits.append(f"final return value bound to '{name}'")
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
        exec(code, namespace)  # noqa: S102  the edited official function is compiled back into its own globals
        new = namespace[self.node.name]
        new.__qualname__ = getattr(self.func, "__qualname__", self.node.name)
        new.__tmg_graft__ = {"label": self.label, "hash": self.hash, "edits": list(self.edits)}
        return new


# installing -------------------------------------------------------------------------------------------
@dataclass
class Installed:
    """Grafted functions swapped into their classes by :func:`install_grafts`, with what is needed to undo it.

    ``info`` maps each attribute name to the compiled function's ``__tmg_graft__`` (label, hash, edits), the
    description the entrypoints log; ``extra_globals`` is the namespace the grafts were compiled with (the runtime).
    """
    targets: list[tuple[object, str, Callable]] = field(default_factory=list)   # (owner, attribute, original function)
    info: dict = field(default_factory=dict)
    extra_globals: dict = field(default_factory=dict)

    def restore(self) -> None:
        """Put every original function back (tests and interactive use); safe to call twice."""
        for owner, attribute, original in reversed(self.targets):
            setattr(owner, attribute, original)
        self.targets = []


def install_grafts(edits: list[tuple[object, str, Graft]], extra_globals: dict) -> Installed:
    """Compile every graft and swap it into its class.

    Args:
        edits: ``(owner, attribute, graft)`` triples; ``graft`` edits ``getattr(owner, attribute)``.
        extra_globals: names the grafted statements use (``{"_tmg": runtime}``), added to each function's globals.

    Returns:
        The :class:`Installed` record whose ``restore`` undoes the swap.
    """
    installed = Installed(extra_globals=extra_globals)
    for owner, attribute, graft in edits:
        compiled = graft.compile(extra_globals)
        installed.targets.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, compiled)
        installed.info[attribute] = compiled.__tmg_graft__
    return installed
