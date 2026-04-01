# Data Sources

## Primary Dataset — World Bank Open Data

| Field | Detail |
|-------|--------|
| **Name** | World Bank Open Data — GDP & Population indicators |
| **Portal URL** | https://data.worldbank.org/ |
| **Availability** | ✅ Publicly available, no authentication required |
| **License** | CC BY 4.0 |
| **Format** | CSV (long format: `Country Name`, `Country Code`, `Series Name`, `Series Code`, `Year`, `Value`) |
| **Missing value marker** | `..` |

### Indicator Codes Used

| Indicator Code | Description | CSV Column (after reshape) |
|----------------|-------------|---------------------------|
| `NY.GDP.MKTP.CD` | GDP (current US$) | `GDP` |
| `NY.GDP.MKTP.PP.CD` | GDP, PPP (current international $) | `PPP_GDP` |
| `SP.POP.TOTL` | Population, total | `Total_Population` |
| `SP.POP.1564.TO` | Population ages 15–64 (working age) | `Working_Age_Population` |

### Local CSV Files

| File | Description | Status |
|------|-------------|--------|
| `Data_GDP_Pop_by_Country_1960.csv` | Full long-format dataset (all series, all years) | ✅ Present in repo |
| `Data_GDP_Pop_by_Country_1960_Countries_only.csv` | Countries only (aggregates removed) — primary input for analysis | ✅ Present in repo |
| `Data_GDP,PPP_Constant 2021_Working_Age_pop_and_Total.csv` | PPP GDP (constant 2021 $), working-age & total population — used as projection baseline | ✅ Present in repo |
| `migration_sim_results.csv` | Output of a sample `run_migration_simulation.py` run | ✅ Present in repo |

## Download / Refresh Instructions

To update the dataset with the latest World Bank figures:

1. Visit https://data.worldbank.org/indicator/NY.GDP.MKTP.CD and select **Download → CSV**.
2. Repeat for `NY.GDP.MKTP.PP.CD`, `SP.POP.TOTL`, and `SP.POP.1564.TO`.
3. Use the World Bank Bulk Download API for programmatic access:
   ```
   https://api.worldbank.org/v2/en/indicator/{CODE}?downloadformat=csv
   ```
4. Replace the corresponding CSV files in the repository root and re-run `pytest -q` to validate.

## No External API Keys Required

All data used by this project is freely downloadable from the World Bank portal with no API key or authentication token.
