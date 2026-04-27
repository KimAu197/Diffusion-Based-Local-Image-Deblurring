#!/usr/bin/env python3
"""Domino stop hook. Resolves `domino_runtime` from the workspace first, then ~/.cursor."""

import json
import sys
from pathlib import Path


def _ensure_domino_path(workspace_root: Path) -> None:
    for base in (
        workspace_root / ".cursor" / "skills" / "domino" / "scripts",
        Path.home() / ".cursor" / "skills" / "domino" / "scripts",
    ):
        if (base / "domino_runtime.py").is_file():
            sys.path.insert(0, str(base))
            return


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({}))
        return 0

    workspace_roots = payload.get("workspace_roots") or []
    if not workspace_roots:
        print(json.dumps({}))
        return 0

    workspace_root = Path(workspace_roots[0]).expanduser().resolve()
    _ensure_domino_path(workspace_root)

    from domino_runtime import load_state, save_state, workspace_path  # noqa: E402

    workspace = workspace_path(workspace_roots[0])
    state = load_state(workspace)
    generation_id = payload.get("generation_id")
    status = payload.get("status")

    if not state.get("active"):
        print(json.dumps({}))
        return 0

    if status != "completed":
        print(json.dumps({}))
        return 0

    if generation_id and generation_id == state.get("last_stop_generation_id"):
        print(json.dumps({}))
        return 0

    if state.get("workflow_status") == "waiting_for_worker":
        print(json.dumps({}))
        return 0

    followup = None
    workflow_status = state.get("workflow_status")
    if workflow_status == "verify_pending":
        followup = (
            "/domino Continue the active Domino workflow and run final verification now. "
            "Read `.cursor/domino-plan.md`, `.cursor/domino-runtime.json`, and the latest task results before acting."
        )
    elif workflow_status == "memory_save_pending":
        followup = (
            "/domino Continue the active Domino workflow and persist memory now. "
            "Write `.cursor/project_state.md` and `.cursor/context_summary.md`, then mark the workflow complete."
        )

    if not followup:
        print(json.dumps({}))
        return 0

    state["last_stop_generation_id"] = generation_id
    save_state(workspace, state)
    print(json.dumps({"followup_message": followup}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
