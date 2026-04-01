# Task 01 — Ingest Data

**Phase:** Coder  
**Role:** Data Engineer  
**Priority:** High  
**Depends on:** —

## Description

Ensure all CSV data files are loaded reliably and consistently through `data_loader.py`. The loader must handle the World Bank long-format CSV (columns: `Country Name`, `Country Code`, `Series Name`, `Series Code`, `Year`, `Value`), strip `..` missing-value markers, and make raw DataFrames available to downstream steps.

## Acceptance Criteria

- [ ] `data_loader.load_and_transform()` loads `Data_GDP_Pop_by_Country_1960_Countries_only.csv` without errors.
- [ ] `data_loader.load_and_transform()` loads `Data_GDP,PPP_Constant 2021_Working_Age_pop_and_Total.csv` without errors.
- [ ] Missing values (`..`) are replaced with `NaN` before any numeric conversion.
- [ ] Year strings in `"YYYY [YRYYYY]"` format are parsed to integers by `parse_year()`.
- [ ] `check_data_completeness()` reports missing-value counts per country, year, and series without raising exceptions.
- [ ] Existing tests in `tests/test_data_loader.py` continue to pass.

## Notes

- Use `config.json` for file paths and column mappings — do not hardcode paths in source files.
- The loader is shared by the dashboard (`gdp_population_analysis.py`) and the simulation entry points (`run_migration_simulation.py`, `run_sim.py`).
