# Task 05 — Test Coverage and Data Validation

**Phase:** Tester  
**Role:** Tester  
**Priority:** Medium  
**Depends on:** 01-ingest-data, 02-clean-data, 03-dashboard, 04-migration-simulation

## Description

Ensure the test suite provides adequate coverage for all major code paths, including data validation against World Bank reference values, transformation correctness, simulation stability, and dashboard callbacks.

## Acceptance Criteria

- [ ] `pytest -q` passes with zero failures across all test modules:
  - `tests/test_data_validation.py`
  - `tests/test_data_loader.py`
  - `tests/test_migration_helpers.py`
  - `tests/test_transform.py`
  - `tests/test_dashboard.py` *(if present)*
- [ ] At least one test verifies GDP/population values against World Bank published statistics for a major economy.
- [ ] Edge cases covered: empty DataFrames, zero population, unknown countries in migration flows, missing series in reshape.
- [ ] All tests are deterministic (no random seeds or order-dependent state).

## Notes

- See `tests/conftest.py` for `sys.path` setup — all new test files must follow the same import pattern.
- Test data should use small, self-contained DataFrames rather than loading production CSV files where possible.
