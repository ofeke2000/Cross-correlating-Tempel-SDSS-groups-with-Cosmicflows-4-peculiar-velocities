"""PostToolUse hook for Edit / Write / MultiEdit / NotebookEdit.

Three checks, one process per edit:

1. **Sign gate (hard rule 2).** An edit to a file that carries the sign conventions
   (``conventions.py``, ``geometry.py``, ``estimators/``) or to one of the gate tests
   re-runs the four sign-gate test files. A failure is fed back to Claude as a block.
2. **Zero-filled velocities (hard rule 5).** Newly written ``fillna(0)`` or
   ``nan_to_num`` is flagged back to Claude as context -- a warning, not a block,
   because a zero fill is legitimate for counts.
3. **Edit log for hard rule 11.** Every edit to ``src/dvcorr/`` or to a notebook is
   timestamped in the session state that ``stop_stale_notebooks.py`` reads.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

from hook_state import load_state, project_dir, repo_relative, save_state

SIGN_GATE_TESTS = (
    "tests/test_geometry.py",
    "tests/test_shell_dipole.py",
    "tests/test_velocity_centered_dipole.py",
    "tests/test_velocity_frame_dipole.py",
)
SIGN_CARRIERS = re.compile(
    r"^src/dvcorr/(conventions\.py|geometry\.py|estimators/.+\.py)$"
)
TRACKED_FOR_NOTEBOOKS = re.compile(r"^(src/dvcorr/.+\.py|(notebooks|intuition)/[^/]+\.ipynb)$")
ZERO_FILL = re.compile(r"fillna\(\s*(value\s*=\s*)?0(\.0*)?\s*[,)]|nan_to_num\(")
SIGN_GATE_TIMEOUT_S = 300
FAILURE_TAIL_LINES = 40


def written_and_replaced_text(tool_name: str, tool_input: dict) -> tuple[str, str]:
    """(text this edit wrote, text it replaced) -- to flag only newly added patterns."""
    if tool_name == "Write":
        return tool_input.get("content", ""), ""
    if tool_name == "Edit":
        return tool_input.get("new_string", ""), tool_input.get("old_string", "")
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits", [])
        return ("\n".join(e.get("new_string", "") for e in edits),
                "\n".join(e.get("old_string", "") for e in edits))
    if tool_name == "NotebookEdit":
        return tool_input.get("new_source", ""), ""
    return "", ""


def run_sign_gate() -> tuple[bool, str]:
    root = project_dir()
    proc = subprocess.run(
        [str(root / ".venv/bin/python"), "-m", "pytest", "-q", "-x", *SIGN_GATE_TESTS],
        cwd=root, capture_output=True, text=True, timeout=SIGN_GATE_TIMEOUT_S,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-FAILURE_TAIL_LINES:])
    return proc.returncode == 0, tail


def main() -> None:
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    rel = repo_relative(path) if path else None
    if rel is None:
        return

    if TRACKED_FOR_NOTEBOOKS.match(rel):
        session_id = payload.get("session_id", "")
        state = load_state(session_id)
        state.setdefault("edits", {})[rel] = time.time()
        save_state(session_id, state)

    messages: list[str] = []
    if rel.endswith((".py", ".ipynb")):
        new, old = written_and_replaced_text(tool_name, tool_input)
        added = set(m.group(0) for m in ZERO_FILL.finditer(new)) - set(
            m.group(0) for m in ZERO_FILL.finditer(old))
        if added:
            messages.append(
                f"Hard rule 5 check: {rel} now contains {sorted(added)}. If this fills a "
                "missing *velocity* (or anything averaged), drop the object instead -- a zero "
                "enters the count without entering the sum. Ignore if it fills a count."
            )

    if SIGN_CARRIERS.match(rel) or rel in SIGN_GATE_TESTS:
        try:
            ok, tail = run_sign_gate()
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok, tail = False, f"sign gate could not run: {exc}"
        if not ok:
            print(json.dumps({
                "decision": "block",
                "reason": (
                    f"Sign gate FAILED after editing {rel} (hard rule 2). Never weaken an "
                    "assertion to make it pass; a wrong-sign dipole on infall is an orientation "
                    "bug. pytest output:\n" + tail
                ),
            }))
            return
        messages.append(f"Sign gate passed after editing {rel}.")

    if messages:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse", "additionalContext": "\n".join(messages)}}))


if __name__ == "__main__":
    main()
