# Task 04 — Migration Simulation

## Summary

Implement and validate the migration-driven population and GDP projection engine.

## Background

The simulation models year-by-year changes in country-level population and GDP under configurable migration flows. Key components:

- `migration_simulation.MigrationSimulator` — core loop; applies population growth projections and migration flows each year.
- `migration_simulation_vectorized.py` — numpy-based `VectorizedMigrationSimulator` and `OptimizedMigrationSimulator` for performance with 200+ countries.
- `scenario_comparison.py` — `ScenarioRunner` for comparing multiple named scenarios side-by-side.
- `run_migration_simulation.py` — example driver that loads repository data and runs a 10-year projection, writing results to `migration_sim_results.csv`.

## Acceptance Criteria

- [ ] `python run_migration_simulation.py` completes without errors and writes `migration_sim_results.csv`.
- [ ] `MigrationSimulator` raises a clear error when required columns are missing from the input DataFrame.
- [ ] Migration flows referencing unknown countries are silently ignored (with a warning logged).
- [ ] `VectorizedMigrationSimulator` produces results numerically equivalent to `MigrationSimulator` for a small test case.
- [ ] Productivity assumptions (`initial_migrant_productivity`, `productivity_step_per_year`) are read from `config.json` and honoured by the simulator.
- [ ] Tests in `tests/test_migration_helpers.py` cover all helper functions and simulator edge cases.

## Relevant Files

- `migration_simulation.py`
- `migration_simulation_vectorized.py`
- `scenario_comparison.py`
- `run_migration_simulation.py`
- `run_sim.py`
- `config.json`
- `tests/test_migration_helpers.py`
- `tests/test_migration_validation.py`
