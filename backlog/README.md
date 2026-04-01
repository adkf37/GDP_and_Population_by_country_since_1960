# Project Background

This project analyzes historical GDP and population data for countries worldwide since 1960, using the World Bank Open Data portal as its primary source. The dataset covers GDP (current USD and PPP-adjusted), total population, and working-age population across 200+ countries and territories. On top of the historical analysis, the project ships a migration-driven projection engine (`MigrationSimulator`) that models how alternative migration flows could reshape population and GDP trajectories over multi-decade horizons. An interactive Dash dashboard allows users to explore derived metrics (GDP per Capita, PPP GDP per Working Age Adult, etc.) filtered by income category, while automated pytest suites validate data quality, transformation logic, and simulation behavior.

## Goals

- Provide a reproducible pipeline that loads, cleans, and transforms World Bank long-format CSV data into analysis-ready wide-format DataFrames.
- Compute and visualize key derived metrics (GDP per Capita, GDP per Working Age Adult, PPP variants) across countries and time.
- Categorize countries by income level and expose the results through an interactive Dash dashboard.
- Implement and validate a year-by-year migration simulation engine that projects population and GDP under configurable migration and productivity assumptions.
- Maintain a comprehensive test suite covering data validation, transformation, simulation correctness, and scenario comparison.

## Success Criteria

- [ ] All CSV data loads without errors and passes World Bank reference-value checks.
- [ ] Derived metrics (GDP per Capita, PPP variants) are calculated correctly and match manual spot-checks.
- [ ] Interactive Dash dashboard launches and renders charts for all supported metrics and income filters.
- [ ] `MigrationSimulator` produces stable, reproducible projections for baseline and shock scenarios.
- [ ] `pytest -q` passes with no failures across all test modules.
- [ ] `backlog/` documentation is complete and ready to guide the Squad Init phase.
