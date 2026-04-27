#!/usr/bin/env python3

import argparse
import json
import re
import sys
import time
from pathlib import Path


RUNTIME_VERSION = 1


def now_ms() -> int:
    return int(time.time() * 1000)


def workspace_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def cursor_dir(workspace: Path) -> Path:
    return workspace / ".cursor"


def tasks_dir(workspace: Path) -> Path:
    return cursor_dir(workspace) / "tasks"


def runtime_path(workspace: Path) -> Path:
    return cursor_dir(workspace) / "domino-runtime.json"


def default_state() -> dict:
    return {
        "version": RUNTIME_VERSION,
        "active": False,
        "workflow_status": "idle",
        "last_worker_role": None,
        "last_task_id": None,
        "last_worker_status": None,
        "last_worker_summary": None,
        "last_modified_files": [],
        "last_subagent_generation_id": None,
        "last_stop_generation_id": None,
        "started_at_ms": None,
        "updated_at_ms": None,
    }


def ensure_workspace(workspace: Path) -> None:
    cursor_dir(workspace).mkdir(parents=True, exist_ok=True)
    tasks_dir(workspace).mkdir(parents=True, exist_ok=True)


def load_state(workspace: Path) -> dict:
    path = runtime_path(workspace)
    if not path.exists():
        return default_state()

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_state()

    state = default_state()
    state.update(parsed if isinstance(parsed, dict) else {})
    if state.get("version") != RUNTIME_VERSION:
        return default_state()
    return state


def save_state(workspace: Path, state: dict) -> None:
    ensure_workspace(workspace)
    state["version"] = RUNTIME_VERSION
    state["updated_at_ms"] = now_ms()
    runtime_path(workspace).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def start_state(workspace: Path) -> dict:
    state = default_state()
    ts = now_ms()
    state.update(
        {
            "active": True,
            "workflow_status": "running",
            "started_at_ms": ts,
            "updated_at_ms": ts,
        }
    )
    save_state(workspace, state)
    return state


def mark_dispatch(workspace: Path, role: str, task_id: str) -> dict:
    state = load_state(workspace)
    if not state.get("active"):
        state = start_state(workspace)
    state.update(
        {
            "active": True,
            "workflow_status": "waiting_for_worker",
            "last_worker_role": role,
            "last_task_id": task_id,
            "last_worker_status": None,
            "last_worker_summary": None,
            "last_modified_files": [],
        }
    )
    save_state(workspace, state)
    return state


def set_status(workspace: Path, status: str) -> dict:
    state = load_state(workspace)
    if not state.get("active"):
        state = start_state(workspace)
    state["workflow_status"] = status
    save_state(workspace, state)
    return state


def complete_state(workspace: Path) -> dict:
    state = load_state(workspace)
    state["active"] = False
    state["workflow_status"] = "completed"
    save_state(workspace, state)
    return state


TASK_STATUS_RE = re.compile(r"^_Status:\s*([A-Za-z\-]+)_\s*$", re.MULTILINE)
TASK_ASSIGNED_RE = re.compile(r"^_Assigned to:\s*([A-Za-z]+)_\s*$", re.MULTILINE)


def task_sort_key(path: Path):
    match = re.search(r"task-(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def parse_task(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    status_match = TASK_STATUS_RE.search(text)
    assigned_match = TASK_ASSIGNED_RE.search(text)
    title = text.splitlines()[0].strip() if text else path.stem
    return {
        "task_id": path.stem,
        "path": str(path),
        "title": title,
        "status": status_match.group(1) if status_match else None,
        "assigned_to": assigned_match.group(1) if assigned_match else None,
    }


def normalize_result(workspace: Path, task_id: str) -> dict:
    path = tasks_dir(workspace) / f"{task_id}.md"
    if not path.exists():
        return {"task_id": task_id, "exists": False}

    text = path.read_text(encoding="utf-8")
    result_marker = "\n## Result\n"
    result_section = ""
    if result_marker in text:
        result_section = text.split(result_marker, 1)[1].strip()

    task = parse_task(path)
    task.update(
        {
            "exists": True,
            "result_section": result_section,
        }
    )
    return task


def next_task(workspace: Path) -> dict:
    directory = tasks_dir(workspace)
    candidates = []
    for path in sorted(directory.glob("task-*.md"), key=task_sort_key):
        task = parse_task(path)
        if task["status"] in {"pending", "revision-needed"}:
            candidates.append(task)
    return {"next_task": candidates[0] if candidates else None, "count": len(candidates)}


def print_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Domino runtime helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--workspace", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--workspace", required=True)

    read_state_parser = subparsers.add_parser("read-state")
    read_state_parser.add_argument("--workspace", required=True)

    dispatch_parser = subparsers.add_parser("mark-dispatch")
    dispatch_parser.add_argument("--workspace", required=True)
    dispatch_parser.add_argument("--role", required=True)
    dispatch_parser.add_argument("--task-id", required=True)

    set_status_parser = subparsers.add_parser("set-status")
    set_status_parser.add_argument("--workspace", required=True)
    set_status_parser.add_argument("--status", required=True)

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("--workspace", required=True)

    next_task_parser = subparsers.add_parser("next-task")
    next_task_parser.add_argument("--workspace", required=True)

    normalize_parser = subparsers.add_parser("normalize-result")
    normalize_parser.add_argument("--workspace", required=True)
    normalize_parser.add_argument("--task-id", required=True)

    args = parser.parse_args()
    workspace = workspace_path(args.workspace)

    if args.command == "ensure":
        ensure_workspace(workspace)
        print_json({"ok": True, "workspace": str(workspace), "runtime_path": str(runtime_path(workspace))})
        return 0

    if args.command == "start":
        print_json(start_state(workspace))
        return 0

    if args.command == "read-state":
        print_json(load_state(workspace))
        return 0

    if args.command == "mark-dispatch":
        print_json(mark_dispatch(workspace, args.role, args.task_id))
        return 0

    if args.command == "set-status":
        print_json(set_status(workspace, args.status))
        return 0

    if args.command == "complete":
        print_json(complete_state(workspace))
        return 0

    if args.command == "next-task":
        print_json(next_task(workspace))
        return 0

    if args.command == "normalize-result":
        print_json(normalize_result(workspace, args.task_id))
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
