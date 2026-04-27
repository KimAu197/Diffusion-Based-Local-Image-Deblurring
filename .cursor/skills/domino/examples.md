# Domino Examples

## Example 1: New Sequential Run

User message:

```text
/domino Build a small evaluation pipeline for dataset X.
```

Expected flow:

1. Domino resolves the workspace root.
2. Domino initializes `.cursor/domino-runtime.json`.
3. Domino dispatches the `Planner` role.
4. Before dispatch, Domino marks:

```bash
python3 ~/.cursor/skills/domino/scripts/domino_runtime.py mark-dispatch --workspace "<workspace-root>" --role "planner" --task-id "plan-001"
```

5. The `subagentStop` hook auto-submits a continuation message.
6. Domino reads the task queue and dispatches the next worker.

## Example 2: Automatic Continuation After Worker Completion

The `subagentStop` hook should emit a follow-up like:

```text
/domino Continue the active Domino workflow in this workspace. A worker subagent just completed successfully. Read `.cursor/domino-plan.md`, `.cursor/domino-runtime.json`, and `.cursor/tasks/` before deciding the next step.
```

Domino should then:

- re-read the runtime files
- inspect the finished task
- decide whether to accept, revise, or dispatch the next worker

## Example 3: Verification Continuation

When all tasks are done, Domino should set:

```bash
python3 ~/.cursor/skills/domino/scripts/domino_runtime.py set-status --workspace "<workspace-root>" --status "verify_pending"
```

The `stop` hook should then emit:

```text
/domino Continue the active Domino workflow and run final verification now. Read `.cursor/domino-plan.md`, `.cursor/domino-runtime.json`, and the latest task results before acting.
```

## Example 4: Memory Save Continuation

After verification passes, Domino should set:

```bash
python3 ~/.cursor/skills/domino/scripts/domino_runtime.py set-status --workspace "<workspace-root>" --status "memory_save_pending"
```

The `stop` hook should then emit:

```text
/domino Continue the active Domino workflow and persist memory now. Write `.cursor/project_state.md` and `.cursor/context_summary.md`, then mark the workflow complete.
```

## Example 5: Blocked Workflow

If a worker is blocked, Domino should not ask the user to run another command manually.

Instead, Domino should ask one question like:

```text
Domino needs one decision before proceeding.

Question: Which dataset split should this pipeline target first?
Options:
  A) validation only
  B) train + validation
Recommendation: A — smaller blast radius for the first pass
```
