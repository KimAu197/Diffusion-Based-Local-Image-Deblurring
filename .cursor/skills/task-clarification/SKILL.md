---
name: task-clarification
description: Clarify vague or under-specified requests into execution-ready plans and prompts before implementation.
argument-hint: [task description]
disable-model-invocation: true
---

# Task Clarification Skill

Transform raw user intent into a clearer execution specification before any implementation begins.

## When to use

Use when the request is:
- broad, ambiguous, or under-specified
- only partially described
- a mix of goals, constraints, and ideas without structure
- something a downstream agent needs a cleaner handoff for

Examples: "help me build an agent for this workflow", "fix this code", "read this repo and tell me what matters", "I want a better prompt for this task"

## When not to use

- The task is already precise and execution-ready
- The user asks for a tiny, direct transformation
- Clarification overhead would exceed its value

---

## Procedure

### Phase 1 — Understand

1. **Restate the goal** — rewrite the request as a precise objective: what to produce, what success looks like, implied scope
2. **Identify task type** — coding / debugging / research / writing / data-analysis / planning / other
3. **Extract available context** — user-stated constraints, env/repo clues, domain hints, implicit expectations
4. **Find gaps** — missing output format, scope limits, audience, tools, language/framework assumptions, brainstorm vs. plan vs. execute
5. **Make assumptions** — don't block on every ambiguity; label assumptions clearly
6. **Gather context lightly** — if tools are available, inspect only what sharpens the task (repo structure, key files, configs); avoid unnecessary exploration

### Phase 2 — Output

7. **Write the execution spec** — objective, scope, constraints, expected output, quality bar, things to avoid
8. **Write the final prompt** — a prompt another model or agent can execute directly

---

## Output format

Scale verbosity to uncertainty level.

### Low uncertainty — compact
```
Goal: ...
Assumptions: ...
Execution Prompt: ...
```
Keep it short. Don't invent complexity.

### Medium / High uncertainty — full
```
Task Type: ...
Goal: ...
Why this task exists: ...
Available Context: ...
Missing Information:
  - ...
Assumptions:
  - ...
Risks / Failure Modes:
  - ...
Recommended Plan:
  1. ...
Execution Spec: ...
Final Execution Prompt: ...
```
For high uncertainty: be explicit about interpretations, identify failure modes, narrow scope before handoff.

---

## Supporting files
- Examples: [examples/](examples/) — see coding.md and research.md for full worked transformations

---

## Quality bar

A good clarification makes the downstream task:
- easier to execute
- less ambiguous
- more scoped
- more testable
- less likely to fail due to misunderstanding
