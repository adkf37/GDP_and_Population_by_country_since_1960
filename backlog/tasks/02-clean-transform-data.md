# Task 02 — Clean and Transform Data

## Summary

Reshape the long-format raw data into a wide per-country/per-year DataFrame and calculate all derived economic metrics.

## Background

The raw CSV is in long format (one row per country/year/series combination). The analysis pipeline needs it in wide format with one row per country/year and separate columns for each series (GDP, PPP_GDP, Total_Population, Working_Age_Population). After reshaping, derived metrics must be computed:

- **GDP per Capita** = GDP / Total_Population
- **GDP per Working Age Adult** = GDP / Working_Age_Population
- **PPP GDP per Capita** = PPP_GDP / Total_Population
- **PPP GDP per Working Age Adult** = PPP_GDP / Working_Age_Population

Countries are then classified into "High Income" (top N% by GDP per Capita, controlled by `config.json`) or "Other".

## Acceptance Criteria

- [ ] `data_loader.reshape_data()` pivots the DataFrame correctly; column names match `series_name_mapping` in `config.json`.
- [ ] `data_loader.calculate_derived_metrics()` produces all four derived columns without division-by-zero errors (zero denominators replaced with `NaN`).
- [ ] Income category column is added with correct "High Income" / "Other" values.
- [ ] `parse_year()` converts `"2022 [YR2022]"` → `2022` (integer) for all standard World Bank year strings.
- [ ] Tests in `tests/test_transform.py` and `tests/test_data_loader.py` cover reshape, metric calculation, and year parsing edge cases.

## Relevant Files

- `data_loader.py`
- `gdp_population_analysis.py`
- `config.json`
- `tests/test_transform.py`
- `tests/test_data_loader.py`
