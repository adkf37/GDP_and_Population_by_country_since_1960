# Project Phases

## Phase 1 — Planner *(current)*

**Goal:** Survey the existing codebase, define the research questions, identify data sources, and produce the `backlog/` documentation that guides all subsequent work.

**Outputs:**
- `backlog/README.md` — project background, goals, success criteria
- `backlog/data_sources.md` — data sources with URLs and availability
- `backlog/phases.md` — this file
- `backlog/tasks/*.md` — discrete task files
- `STATUS.md` — current objective and phase

**Exit Criteria:** All acceptance criteria in the Planner task are met and `STATUS.md` reflects the transition to Squad Init.

---

## Phase 2 — Squad Init

**Goal:** Assemble the agent team based on the plan and assign roles to discrete tasks.

**Outputs:**
- `.squad/team.md` — agent roster with role assignments
- `.squad/routing.md` — routing rules for task dispatch

**Exit Criteria:** Each task in `backlog/tasks/` has an assigned agent role; `.squad/team.md` and `.squad/routing.md` exist.

---

## Phase 3 — Coder

**Goal:** Implement all tasks in `backlog/tasks/` according to the plan. Squad routes work to specialist agents (Data Engineer, Statistician, etc.).

**Outputs:**
- Updated Python source files (`data_loader.py`, `gdp_population_analysis.py`, `migration_simulation.py`, etc.)
- New or updated tests under `tests/`
- Updated `requirements.txt` if new dependencies are added

**Exit Criteria:** All task files in `backlog/tasks/` are marked complete; `pytest -q` passes.

---

## Phase 4 — Reconciler *(if dual-coder mode was used)*

**Goal:** Compare outputs from parallel Coder agents, resolve conflicts, and produce a unified implementation.

**Outputs:**
- Reconciled source files
- Diff summary in `.squad/decisions.md`

**Exit Criteria:** No merge conflicts remain; tests pass on the reconciled codebase.

---

## Phase 5 — Reviewer

**Goal:** Critical review of the full implementation — correctness, test coverage, documentation accuracy, and final polish.

**Outputs:**
- Review comments addressed
- `FEEDBACK.md` updated with any outstanding concerns
- Final `STATUS.md` update marking the project complete

**Exit Criteria:** Reviewer sign-off; all acceptance criteria from `backlog/README.md` are satisfied.
