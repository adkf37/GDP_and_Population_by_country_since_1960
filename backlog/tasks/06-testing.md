# Task 06 — Testing & Coverage

## Summary

Ensure the test suite is comprehensive, all tests pass, and coverage gaps are filled.

## Background

The existing test files are:

| File | Scope |
|------|-------|
| `tests/test_data_loader.py` | `data_loader.py` unit and integration tests |
| `tests/test_data_validation.py` | GDP/population values against World Bank published statistics |
| `tests/test_migration_helpers.py` | Migration helper functions and simulator edge cases |
| `tests/test_migration_validation.py` | Migration flow realism checks |
| `tests/test_transform.py` | Derived metric calculation tests |
| `tests/test_dashboard.py` | Dash callback and layout tests |

## Acceptance Criteria

- [ ] `pytest -q` reports zero failures.
- [ ] All critical code paths in `data_loader.py`, `migration_simulation.py`, and `gdp_population_analysis.py` have at least one test.
- [ ] Edge cases are covered: empty DataFrames, all-NaN columns, unknown countries in migration flows, zero-population countries.
- [ ] `tests/conftest.py` correctly adds the repo root to `sys.path` so all imports work without package installation.
- [ ] No test imports are broken by refactoring done in other tasks.

## Relevant Files

- `tests/` (all files)
- `conftest.py`
