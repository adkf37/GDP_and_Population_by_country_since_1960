# Task 03 — Interactive Dashboard

**Phase:** Coder  
**Role:** Statistician  
**Priority:** Medium  
**Depends on:** 02-clean-data

## Description

Build and validate the interactive Dash dashboard (`gdp_population_analysis.py`) that lets users explore derived metrics across countries and income categories.

## Acceptance Criteria

- [ ] Dashboard launches at `http://127.0.0.1:8050/` when running `python gdp_population_analysis.py`.
- [ ] Metric dropdown populates with all supported options: `GDP_per_Capita`, `PPP_GDP_per_Capita`, `GDP_per_Working_Age_Adult`, `PPP_GDP_per_Working_Age_Adult`.
- [ ] Income-category checkboxes (`High Income`, `Other`) correctly filter the scatter plot.
- [ ] Hovering over a data point shows the country name.
- [ ] Static Matplotlib scatter plot (GDP per Capita vs. GDP per Working Age Adult) is generated without errors.
- [ ] Correlation between total GDP and GDP per working-age adult is printed to stdout.
- [ ] Existing tests in `tests/test_dashboard.py` pass.

## Notes

- All configuration (file paths, column names) must be read from `config.json`.
- Do not hardcode the income threshold — use `config.json` → `income_percentile_threshold`.
