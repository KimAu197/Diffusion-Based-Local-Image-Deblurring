#!/usr/bin/env python3
"""Domino subagentStop hook. Resolves `domino_runtime` from the workspace first, then ~/.cursor."""

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

    if state.get("workflow_status") != "waiting_for_worker":
        print(json.dumps({}))
        return 0

    if status != "completed":
        state["last_worker_status"] = status
        save_state(workspace, state)
        print(json.dumps({}))
        return 0

    if generation_id and generation_id == state.get("last_subagent_generation_id"):
        print(json.dumps({}))
        return 0

    summary = (payload.get("summary") or "").strip()
    task = (payload.get("task") or "").strip()
    modified_files = payload.get("modified_files") or []

    state["workflow_status"] = "running"
    state["last_worker_status"] = status
    state["last_worker_summary"] = summary[:4000] if summary else None
    state["last_modified_files"] = modified_files
    state["last_subagent_generation_id"] = generation_id
    save_state(workspace, state)

    followup = (
        "/domino Continue the active Domino workflow in this workspace. "
        "A worker subagent just completed successfully. "
        "Read `.cursor/domino-plan.md`, `.cursor/domino-runtime.json`, and any updated `.cursor/tasks/` files before deciding the next step."
    )

    if task:
        followup += f"\n\nCompleted worker task: {task}"
    if summary:
        followup += f"\n\nWorker summary:\n{summary[:1500]}"

    print(json.dumps({"followup_message": followup}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
