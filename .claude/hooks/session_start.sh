#!/usr/bin/env bash
# SessionStart hook: activate the project venv for every Bash call of the session,
# and warn when that venv cannot import dvcorr (e.g. after an OS upgrade removed the
# Python it was built on -- what happened on the 22.04 -> 24.04 upgrade).
venv="$CLAUDE_PROJECT_DIR/.venv"

if [ -n "$CLAUDE_ENV_FILE" ] && [ -f "$venv/bin/activate" ]; then
  echo "source \"$venv/bin/activate\"" >> "$CLAUDE_ENV_FILE"
fi

if ! "$venv/bin/python" -c "import dvcorr, pytest" 2>/dev/null; then
  msg="The project venv ($venv) cannot import dvcorr/pytest. It is probably built on a Python that no longer exists. Rebuild it: mv .venv .venv-old && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  printf '{"systemMessage": "%s", "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "%s"}}\n' "$msg" "$msg"
fi
exit 0
