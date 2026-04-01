# Task 03 — Exploratory Analysis & Dashboard

## Summary

Provide an interactive Dash/Plotly web dashboard that lets researchers explore the cleaned GDP and population data.

## Background

`gdp_population_analysis.py` is the entry point for the dashboard. It:

1. Loads and transforms data via `data_loader.load_and_transform()`.
2. Classifies countries by income tier.
3. Creates a Dash app with:
   - A metric dropdown (GDP per Capita, PPP GDP per Capita, GDP per Working Age Adult, PPP GDP per Working Age Adult).
   - Income category checkboxes (High Income / Other).
   - A scatter plot of selected metric vs. Total Population, coloured by income category.

## Acceptance Criteria

- [ ] `python gdp_population_analysis.py` starts without errors and the Dash server is accessible at `http://127.0.0.1:8050/`.
- [ ] Changing the metric dropdown updates the scatter plot without page reload.
- [ ] Toggling income category checkboxes correctly filters the displayed countries.
- [ ] Hovering over a data point shows the country name.
- [ ] Dashboard callback tests in `tests/test_dashboard.py` pass.

## Relevant Files

- `gdp_population_analysis.py`
- `data_loader.py`
- `config.json`
- `tests/test_dashboard.py`
