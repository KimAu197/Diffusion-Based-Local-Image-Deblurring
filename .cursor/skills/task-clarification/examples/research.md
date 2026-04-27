# Example: Vague Research Task

## Raw request
"Read this repo and tell me what matters"

---

## Clarified output

**Task Type:** research

**Goal:** Produce a concise orientation of the repo — its purpose, structure, key entry points, and non-obvious decisions — for someone about to work in it.

**Available Context:**
- Repo is available locally
- User is likely onboarding or about to start a task

**Missing Information:**
- What will the user do next? (fix a bug, add a feature, understand architecture?)
- Any specific area of focus? (backend, frontend, data layer, infra?)
- How much depth? (quick overview vs. deeper dive?)

**Assumptions:**
- User wants an orientation, not exhaustive documentation
- Medium depth: enough to navigate confidently, not a full audit
- No specific area focus — cover the whole repo at a high level

**Risks:**
- Over-summarizing (losing what actually matters)
- Under-summarizing (listing files instead of explaining purpose)

**Recommended Plan:**
1. Read README and top-level structure
2. Identify entry points (main, index, CLI, API surface)
3. Note key dependencies and what they signal about the architecture
4. Flag non-obvious or important decisions (unusual patterns, TODOs, legacy areas)
5. Summarize in plain language

**Final Execution Prompt:**
> Read this repo and produce a concise orientation. Cover: what it does, how it's structured, key entry points, important dependencies, and anything non-obvious a new contributor should know. Skip exhaustive file listings. Prioritize what actually matters for navigating and working in this codebase.
