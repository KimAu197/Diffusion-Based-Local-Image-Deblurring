# Domino Reference

## Role Mapping

The old slash-command architecture maps to internal Domino roles like this:

| Old command | New role inside Domino |
|-------------|------------------------|
| `/domino` | Domino skill |
| `/orchestrator` | Planner |
| `/orchestrator-parallel` | ParallelPlanner |
| `/agent-executor` | Executor |
| `/agent-debugger` | Debugger |
| `/agent-reviewer` | Reviewer |
| `/verify` | Verifier |
| `/memory-load` | Domino startup restore step |
| `/memory-save` | MemorySaver |

## Standard Task Schema

Every standard task file should keep this shape:

```markdown
# Task {NNN}: {short title}
_Created by: Domino Planner_
_Assigned to: {Executor | Debugger | Reviewer}_
_Status: pending_

## Objective
One sentence: what should be true when this task is done?

## Context
- Relevant files: [exact paths]
- Current state: [what exists now]
- Dependencies: [task IDs, or "none"]
- Domino assumptions: [list active assumptions, or "none"]
- User decisions: [list binding answers, or "none"]

## Instructions
Precise, step-by-step instructions.

## Acceptance Criteria
- [ ] [specific, checkable condition]

## Output
- Modified files: [list]
- Result summary: write under `## Result`
```

## Worker Contracts

### Planner

Use for sequential, repair, and review-only segments.

Required output:

- Write standard task specs to `.cursor/tasks/`
- Keep assumptions and user decisions explicit
- Do not ask the user to manually trigger a worker if Domino can dispatch it next

### ParallelPlanner

Use for parallel or hybrid wave planning.

Required output:

- Write standard task specs to `.cursor/tasks/`
- Write `.cursor/parallel-plan.md`
- Only parallelize work with explicit interface clarity
- Carry assumptions and user decisions into both task specs and wave plan

### Executor

Use this contract:

1. Read the task file first.
2. Read `Domino assumptions` and `User decisions`.
3. Treat them as binding execution constraints.
4. Stay in scope.
5. Write a result section including:
   - `Files changed`
   - `Acceptance criteria`
   - `Constraint check`

### Debugger

Use this contract:

1. Diagnose before fixing.
2. Read `Domino assumptions` and `User decisions`.
3. Do not apply a fix that violates those constraints.
4. Write a result section including:
   - `Root cause`
   - `Fix applied`
   - `Verification`
   - `Constraint check`

### Reviewer

Use this contract:

1. Read the task file and the completed result.
2. Read actual files, not just the reported summary.
3. Check acceptance criteria explicitly.
4. Check whether `Domino assumptions` and `User decisions` were satisfied.
5. Write a result section including:
   - `Verdict`
   - `Criteria review`
   - `Additional findings`
   - `Constraint check`

### Verifier

Use this contract:

1. Re-read the actual changed files.
2. Compare them to the goal and plan.
3. Check for correctness, completeness, references, and consistency.
4. If there are issues, create repair follow-up work through the Planner role.
5. If everything passes, report clearly that verification passed.

### MemorySaver

Use this contract:

1. Scan the actual workspace.
2. Write `.cursor/project_state.md`.
3. Write `.cursor/context_summary.md`.
4. Keep both files precise and self-sufficient.

## Runtime File Contract

Domino relies on `.cursor/domino-runtime.json` to coordinate hooks and follow-up turns.

Expected fields:

```json
{
  "version": 1,
  "active": true,
  "workflow_status": "running",
  "last_worker_role": null,
  "last_task_id": null,
  "last_subagent_generation_id": null,
  "last_stop_generation_id": null
}
```

## Hook Behavior

### subagentStop

Trigger a follow-up only when:

- the runtime state is active
- `workflow_status` is `waiting_for_worker`
- the subagent status is `completed`
- the generation id has not already been processed

### stop

Trigger a follow-up only when:

- the runtime state is active
- `workflow_status` is `verify_pending` or `memory_save_pending`
- the stop event status is `completed`
- the generation id has not already been processed

## Parallel Note

Parallel execution is phase-two quality, not phase-one quality.

The first stable target is:

- sequential
- repair
- review-only
- final verify
- memory save

Only expand the parallel path once the sequential hook loop is reliable.
