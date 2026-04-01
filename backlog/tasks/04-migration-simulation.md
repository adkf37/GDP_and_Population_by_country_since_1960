# Task 04 — Migration Simulation Engine

**Phase:** Coder  
**Role:** Statistician  
**Priority:** Medium  
**Depends on:** 02-clean-data

## Description

Validate and extend the `MigrationSimulator` and related helpers (`dataframe_to_projection_dict`, `migration_flows_from_dataframe`) so that multi-decade population and GDP projections are correct, robust, and reproducible.

## Acceptance Criteria

- [ ] `MigrationSimulator` raises a clear error when required columns (`Country`, `Year`, `Total_Population`, `GDP`) are missing from the input DataFrame.
- [ ] Migration flows referencing unknown countries are silently ignored (no `KeyError`).
- [ ] Division by zero is prevented when computing per-capita GDP after migration flows reduce a country's population to zero.
- [ ] `dataframe_to_projection_dict` correctly converts a projection DataFrame to `Dict[Tuple[str, int], float]`.
- [ ] `migration_flows_from_dataframe` correctly converts a flows DataFrame to `Dict[int, List[Tuple[str, str, float]]]`.
- [ ] `run_migration_simulation.py` completes without errors when run against the repository's CSV files.
- [ ] All existing tests in `tests/test_migration_helpers.py` pass.
- [ ] Productivity parameters (`init_prod`, `prod_step`) are read from `config.json` → `simulation` section.

## Notes

- The vectorized implementations in `migration_simulation_vectorized.py` should produce results consistent with the reference `MigrationSimulator` for the same inputs.
- See `scenario_comparison.py` for multi-scenario orchestration helpers.
