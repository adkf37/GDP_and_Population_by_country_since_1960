# Configuration Schema

This document describes the `config.json` configuration file.

## Data Loading

| Key | Type | Description |
|-----|------|-------------|
| `data_file_path` | string | Path to the main CSV data file (World Bank format) |
| `high_income_percentile_threshold` | number | Percentile (0-100) for classifying "High Income" countries. Countries with GDP per capita at or above this percentile are classified as High Income. Default: 80 |

## Data Transformation

These settings control how the long-format World Bank CSV is pivoted to wide format.

| Key | Type | Description |
|-----|------|-------------|
| `pivot_index_columns` | array | Columns to use as the index when pivoting. Default: `["Country Name", "Year", "Country Code"]` |
| `pivot_columns_column` | string | Column whose values become new column names after pivoting. Default: `"Series Name"` |
| `pivot_values_column` | string | Column containing the data values. Default: `"Value"` |

## Series Name Mapping

The `series_name_mapping` object maps World Bank series names to internal column names:

| World Bank Series Name | Internal Name | Description |
|------------------------|---------------|-------------|
| `GDP (current US$)` | `GDP` | Gross Domestic Product in current US dollars |
| `GDP, PPP (current international $)` | `PPP_GDP` | GDP adjusted for Purchasing Power Parity |
| `Population, total` | `Total_Population` | Total population count |
| `Population, ages 15-64, total` | `Working_Age_Population` | Working-age population (15-64 years) |

## Simulation Settings

The `simulation` object contains parameters for the migration simulation engine:

| Key | Type | Description |
|-----|------|-------------|
| `initial_migrant_productivity` | number | Initial productivity factor for migrants (0.0-1.0). A value of 0.5 means migrants initially contribute 50% of native productivity. Default: 0.5 |
| `productivity_step_per_year` | number | Annual increase in migrant productivity. A value of 0.1 means productivity increases by 10 percentage points per year until reaching 1.0. Default: 0.1 |
| `default_projection_years` | number | Default number of years to project forward in simulations. Default: 10 |

## World Bank Indicator Codes

The data is sourced from World Bank Open Data. Relevant indicator codes:

- `NY.GDP.MKTP.CD` - GDP (current US$)
- `NY.GDP.MKTP.PP.CD` - GDP, PPP (current international $)
- `SP.POP.TOTL` - Population, total
- `SP.POP.1564.TO` - Population, ages 15-64, total

Data source: https://data.worldbank.org/
