# Task 01 — Ingest Raw Data

## Summary

Load the raw World Bank CSV files into memory and make them available to the rest of the pipeline.

## Background

The repository ships two primary CSV sources:

1. `Data_GDP_Pop_by_Country_1960_Countries_only.csv` — country-level records (regional aggregates removed), used by the exploratory analysis dashboard.
2. `Data_GDP,PPP_Constant 2021_Working_Age_pop_and_Total.csv` — PPP-constant series used as the migration simulation baseline.

Both use the World Bank long format (`Country Name`, `Country Code`, `Series Name`, `Series Code`, `Year`, `Value`), where `Value` may be `..` for missing data.

## Acceptance Criteria

- [ ] `data_loader.load_and_transform()` reads the configured CSV path from `config.json` without errors.
- [ ] All rows with `Value == ".."` are correctly converted to `NaN` (not dropped).
- [ ] The loaded DataFrame has at minimum the columns: `Country Name`, `Country Code`, `Year`, `Series Name`, `Value`.
- [ ] A unit test in `tests/test_data_loader.py` confirms successful loading and that `..` values become `NaN`.

## Relevant Files

- `data_loader.py`
- `config.json`
- `tests/test_data_loader.py`
