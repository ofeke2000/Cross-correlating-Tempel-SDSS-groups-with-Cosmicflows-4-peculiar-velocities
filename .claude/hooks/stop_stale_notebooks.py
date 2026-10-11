"""Stop hook enforcing hard rule 11: a changed notebook is re-executed before "done".

A notebook is *affected* when Claude edited it directly this session, or edited a
``src/dvcorr`` module that the notebook imports, directly or transitively (the import
graph is read from the source with ``ast``). An affected notebook is *stale* while its
file is not newer than that edit: ``nbconvert --execute --inplace`` rewrites the file
on success, so a fresh mtime is the proof of a re-run.

When stale notebooks remain, the first stop is blocked once with the list. Claude then
either re-runs them or -- for a long full-catalog run that needs the user's go-ahead --
names them as stale in its final message. The same (notebook, edit) pair is never
reported twice, so a deliberate deferral does not nag on every later turn.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

from hook_state import load_state, project_dir, save_state

NOTEBOOK_DIRS = ("notebooks", "intuition")
MAGIC_LINE = re.compile(r"^\s*[%!?]")
MTIME_SLACK_S = 1.0


def module_name(rel: str) -> str:
    """'src/dvcorr/pipeline/x.py' -> 'dvcorr.pipeline.x'; '__init__.py' -> its package."""
    parts = Path(rel).with_suffix("").parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_dvcorr_modules(tree: ast.AST, package: str, known: set[str]) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "dvcorr")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod.split(".")[0] != "dvcorr":
                continue
            found.add(mod)
            found.update(f"{mod}.{a.name}" for a in node.names if f"{mod}.{a.name}" in known)
    with_parents = set()
    for mod in found:
        parts = mod.split(".")
        with_parents.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return with_parents & known


def dvcorr_import_graph(root: Path) -> dict[str, set[str]]:
    files = {module_name(p.relative_to(root).as_posix()): p
             for p in (root / "src/dvcorr").rglob("*.py")}
    known = set(files)
    graph = {}
    for mod, path in files.items():
        package = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
        try:
            graph[mod] = imported_dvcorr_modules(ast.parse(path.read_text()), package, known)
        except SyntaxError:
            graph[mod] = set()
    return graph


def notebook_imports(path: Path, known: set[str]) -> set[str]:
    try:
        cells = json.loads(path.read_text()).get("cells", [])
    except (OSError, ValueError):
        return set()
    found: set[str] = set()
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        src = "\n".join("" if MAGIC_LINE.match(l) else l for l in src.splitlines())
        try:
            found |= imported_dvcorr_modules(ast.parse(src), "", known)
        except SyntaxError:
            continue
    return found


def closure(start: set[str], graph: dict[str, set[str]]) -> set[str]:
    seen, todo = set(), list(start)
    while todo:
        mod = todo.pop()
        if mod not in seen:
            seen.add(mod)
            todo.extend(graph.get(mod, ()))
    return seen


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("stop_hook_active"):
        return
    session_id = payload.get("session_id", "")
    state = load_state(session_id)
    edits: dict[str, float] = state.get("edits", {})
    if not edits:
        return

    root = project_dir()
    module_edits = {module_name(r): t for r, t in edits.items() if r.startswith("src/dvcorr/")}
    graph = dvcorr_import_graph(root) if module_edits else {}
    reported = set(state.get("reported", []))

    stale: list[tuple[str, float, str]] = []
    for nb in sorted(p for d in NOTEBOOK_DIRS for p in (root / d).glob("*.ipynb")):
        rel = nb.relative_to(root).as_posix()
        causes = {rel: edits[rel]} if rel in edits else {}
        if module_edits:
            deps = closure(notebook_imports(nb, set(graph)), graph)
            causes.update({m: t for m, t in module_edits.items() if m in deps})
        if not causes:
            continue
        cause, edited_at = max(causes.items(), key=lambda kv: kv[1])
        key = f"{rel}@{edited_at:.0f}"
        if nb.stat().st_mtime <= edited_at + MTIME_SLACK_S and key not in reported:
            stale.append((rel, edited_at, cause))
            reported.add(key)

    if not stale:
        return
    state["reported"] = sorted(reported)
    save_state(session_id, state)
    lines = "\n".join(f"  - {nb}  (changed: {cause})" for nb, _, cause in stale)
    print(json.dumps({
        "decision": "block",
        "reason": (
            "Hard rule 11: these notebooks are stale -- their outputs predate this session's "
            f"edits:\n{lines}\nRe-execute each with `jupyter nbconvert --to notebook --execute "
            "--inplace <path>`. If a run is long (full catalog), ask the user first instead, and "
            "name the notebooks as stale in your final message."
        ),
    }))


if __name__ == "__main__":
    main()
