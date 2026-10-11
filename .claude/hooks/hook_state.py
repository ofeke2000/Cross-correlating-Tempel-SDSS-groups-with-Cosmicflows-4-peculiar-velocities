"""Per-session state shared by the project's Claude Code hooks.

``post_edit.py`` records, for every file Claude edits, the time of the edit;
``stop_stale_notebooks.py`` reads that record at the end of a turn to decide which
notebooks are stale under hard rule 11. The state lives outside the repository
(one JSON file per session under the system temp dir), so nothing here is committed.

State layout::

    {"edits": {"<repo-relative path>": <unix time of last edit>},
     "reported": ["<notebook>@<edit time>", ...]}
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

STATE_DIR = Path(tempfile.gettempdir()) / "dvcorr-claude-hooks"


def project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())).resolve()


def state_path(session_id: str) -> Path:
    return STATE_DIR / f"{session_id or 'unknown'}.json"


def load_state(session_id: str) -> dict:
    try:
        return json.loads(state_path(session_id).read_text())
    except (OSError, ValueError):
        return {"edits": {}, "reported": []}


def save_state(session_id: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(session_id).write_text(json.dumps(state))


def repo_relative(path: str) -> str | None:
    """Repo-relative POSIX path of ``path``, or None when it lies outside the repo."""
    try:
        return Path(path).resolve().relative_to(project_dir()).as_posix()
    except ValueError:
        return None
