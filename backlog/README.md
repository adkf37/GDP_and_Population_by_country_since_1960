# Project Background

This project analyzes historical GDP and population data for countries worldwide from 1960 onwards, using data sourced from the World Bank Open Data portal. It provides a data pipeline that ingests long-format World Bank CSV exports, reshapes and cleans the data into a wide per-country/per-year format, calculates derived metrics (GDP per Capita, PPP-adjusted figures, working-age variants), and exposes an interactive Dash dashboard for exploratory analysis. On top of the historical analysis, the project includes a migration-driven projection engine (`MigrationSimulator`) and vectorized variants that allow researchers to model how alternative international migration flows might reshape population distributions and GDP trajectories for individual countries through the mid-21st century.

## Goals

- **Ingest & reshape raw World Bank data** — load long-format CSV exports and pivot them to a clean wide-format DataFrame with consistent column names.
- **Calculate derived economic metrics** — compute GDP per Capita, GDP per Working Age Adult, and PPP variants; classify countries by income tier.
- **Provide interactive exploratory analysis** — deliver a Dash/Plotly dashboard that lets users filter by income category and switch between metrics.
- **Enable migration scenario modelling** — simulate year-by-year population and GDP trajectories under configurable migration flow assumptions using `MigrationSimulator` and its vectorized equivalents.
- **Ensure correctness through automated testing** — maintain a pytest suite covering data transformation, migration helpers, dashboard callbacks, and data validation against published World Bank statistics.

## Success Criteria

- All CSV data files load without errors via `data_loader.load_and_transform()`.
- Derived metrics (GDP per Capita, PPP GDP per Working Age Adult, etc.) are calculated correctly for all country-year rows with sufficient data.
- The Dash dashboard starts without errors and renders scatter plots for every supported metric.
- `MigrationSimulator` completes a 10-year projection without errors on the bundled dataset.
- `pytest -q` passes with no failures.
- `backlog/` folder is fully populated and `STATUS.md` reflects the current project phase.
