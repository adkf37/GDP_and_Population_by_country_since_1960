# Project Phases

## Phase 1 — Planner

**Goal:** Survey the repository, define research questions, and produce the backlog.

**Outputs:**
- `backlog/README.md` with project background, goals, and success criteria
- `backlog/data_sources.md` with all data sources and availability status
- `backlog/phases.md` (this file)
- `backlog/tasks/` with one file per discrete task
- `STATUS.md` updated with current objective

**Status:** ✅ Complete

---

## Phase 2 — Squad Init

**Goal:** Initialise the agent team based on the plan and assign roles.

**Roles:**
- **Lead** — oversees overall direction and coordinates between agents
- **Data Engineer** — owns data ingestion, transformation, and `data_loader.py`
- **Statistician** — owns derived metric calculations and migration projection models
- **Tester** — owns the `tests/` suite; adds missing coverage and validates correctness
- **Scribe** — owns documentation (`README.md`, `CONFIG_SCHEMA.md`, docstrings)
- **Ralph** — ad-hoc utility work (performance profiling, misc fixes)

**Outputs:**
- `.squad/team.md` with role assignments
- `.squad/routing.md` with routing rules for specialist agents

**Status:** ⬜ Pending

---

## Phase 3 — Coder

**Goal:** Implement all tasks listed in `backlog/tasks/` according to priorities.

**Key activities:**
- Complete any remaining TODO items from `TODO.md`
- Add example outputs / dashboard screenshots (last open TODO item)
- Extend test coverage for edge cases not yet covered
- Refine migration simulation for realism and performance

**Status:** ⬜ Pending

---

## Phase 4 — Reconciler

**Goal:** If dual-coder outputs exist, compare them and merge the best of each.

**Activities:**
- Diff outputs from parallel Coder agents (if applicable)
- Resolve conflicts and produce a unified implementation
- Confirm all tests still pass after merge

**Status:** ⬜ Pending (only needed if dual Coders are used)

---

## Phase 5 — Reviewer

**Goal:** Critical review, fact-checking, final polish before merge.

**Activities:**
- Re-run `pytest -q` and confirm zero failures
- Review all changed files for correctness and style consistency
- Update `STATUS.md` to reflect completion
- Ensure `README.md` is accurate and up-to-date

**Status:** ⬜ Pending
