# Task 02 — Clean and Transform Data

**Phase:** Coder  
**Role:** Data Engineer  
**Priority:** High  
**Depends on:** 01-ingest-data

## Description

Reshape the World Bank long-format CSV into a wide-format DataFrame (one row per country/year, one column per indicator) and calculate all derived metrics required by the dashboard and simulation engine.

## Acceptance Criteria

- [ ] `reshape_data()` (or equivalent) correctly pivots long-format rows into wide-format columns using the mappings in `config.json`.
- [ ] Derived metrics are calculated without division-by-zero errors:
  - `GDP_per_Capita = GDP / Total_Population`
  - `GDP_per_Working_Age_Adult = GDP / Working_Age_Population`
  - `PPP_GDP_per_Capita = PPP_GDP / Total_Population`
  - `PPP_GDP_per_Working_Age_Adult = PPP_GDP / Working_Age_Population`
- [ ] Countries are categorized into `High Income` / `Other` using the percentile threshold from `config.json`.
- [ ] All existing tests in `tests/test_transform.py` and `tests/test_data_loader.py` continue to pass.
- [ ] Zero-population rows produce `NaN` (not `Inf`) for per-capita metrics.

## Notes

- The percentile threshold for "High Income" is configurable in `config.json` (`income_percentile_threshold`).
- Ensure `clean_and_transform_data()` in `gdp_population_analysis.py` and `calculate_derived_metrics()` in `data_loader.py` stay in sync.
