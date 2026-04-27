# Example: Vague Coding Task

## Raw request
"Fix my auth, it's broken"

---

## Clarified output

**Task Type:** debugging

**Goal:** Identify and fix the authentication failure in the codebase.

**Available Context:**
- Auth-related files visible in repo
- "Broken" implies a regression or new failure, not a missing feature

**Missing Information:**
- What specific failure? (login fails, token invalid, redirect loop, 401 on protected routes?)
- Which auth method? (JWT, session, OAuth?)
- When did it break? (after a change, always, intermittently?)
- Framework and language?

**Assumptions:**
- Web app with session or token-based auth
- Local repo is available for inspection
- "Broken" = something that previously worked is now failing

**Risks:**
- Fixing symptoms without finding the root cause
- Auth changes can introduce security regressions — keep changes narrow

**Recommended Plan:**
1. Inspect auth entry points (middleware, login handler, token validation)
2. Reproduce the failure with a specific case
3. Locate the regression (recent change, config, dependency update)
4. Fix narrowly, test the specific failure path

**Final Execution Prompt:**
> Inspect the authentication system in this repo. Look at middleware, login handlers, and token validation. Identify why auth is failing — look for recent changes, misconfiguration, or broken assumptions. Fix the root cause narrowly. Do not refactor surrounding code. Report what you found and what you changed.
