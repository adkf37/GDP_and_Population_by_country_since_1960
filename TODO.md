# TODO: Codebase Improvements

## Data Validation & Quality

- [x] **Add data source validation tests** - Verify GDP/population values match World Bank published statistics for major economies (see `tests/test_data_validation.py`)
- [x] **Add data completeness checks** - Report which countries/years have missing values (`..`) before analysis (see `data_loader.check_data_completeness()`)
- [x] **Validate year parsing** - The `Year` column contains `"2022 [YR2022]"` format; parsing handled by `data_loader.parse_year()`

## Code Quality

- [x] **Fix inconsistent column expectations** - `run_migration_simulation.py` now uses `data_loader.prepare_for_simulation()` for proper column transformation
- [x] **Add type hints** - `gdp_population_analysis.py` now has type annotations
- [x] **Extract common data loading** - Created `data_loader.py` with shared utilities
- [x] **Handle division by zero** - `clean_and_transform_data()` and `data_loader.calculate_derived_metrics()` now replace zero with NaN before division

## Testing

- [x] **Increase test coverage for dashboard** - Tests in `tests/test_dashboard.py` for `create_dashboard()` and Dash callbacks
- [x] **Add integration test** - End-to-end test in `tests/test_data_loader.py::TestIntegration`
- [x] **Test edge cases in reshape_data()** - Tests for missing series, duplicates, malformed values in `tests/test_data_loader.py::TestReshapeLongToWide`

## Migration Simulation

- [x] **Validate migration flow realism** - UN migration statistics reference data and validation tests in `tests/test_migration_validation.py`
- [x] **Add scenario comparison** - Created `scenario_comparison.py` with `ScenarioRunner` for multi-scenario comparison
- [x] **Parameterize productivity assumptions** - `init_prod` and `prod_step` now configurable in `config.json` under `simulation` section

## Documentation

- [x] **Document data source** - README now specifies World Bank as the data source with indicator codes
- [ ] **Add example outputs** - Show sample dashboard screenshots or simulation result summaries
- [x] **Document config.json schema** - Created `CONFIG_SCHEMA.md` with full documentation

## Performance

- [x] **Optimize simulation for large country sets** - Created `migration_simulation_vectorized.py` with `VectorizedMigrationSimulator` and `OptimizedMigrationSimulator` using numpy arrays
