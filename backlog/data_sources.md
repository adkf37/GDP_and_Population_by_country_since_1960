# Data Sources

## Primary Source — World Bank Open Data

| File | Description | Availability |
|------|-------------|--------------|
| `Data_GDP_Pop_by_Country_1960.csv` | Raw long-format export: GDP (current US$), GDP PPP, Total Population, Working-Age Population for all World Bank member countries from 1960. | ✅ Present in repo |
| `Data_GDP_Pop_by_Country_1960_Countries_only.csv` | Subset of the above with aggregate/regional rows removed; used as the primary input to the analysis pipeline. | ✅ Present in repo |
| `Data_GDP,PPP_Constant 2021_Working_Age_pop_and_Total.csv` | GDP PPP (constant 2021 international $), Working-Age Population, Total Population — used as the baseline for migration projections. | ✅ Present in repo |
| `Data_GDP_Pop_by_Country_1960.xlsx` | Excel copy of the raw long-format data (same content as the CSV counterpart). | ✅ Present in repo |

### World Bank Indicator Codes

| Indicator Code | Description |
|----------------|-------------|
| `NY.GDP.MKTP.CD` | GDP (current US$) |
| `NY.GDP.MKTP.PP.CD` | GDP, PPP (current international $) |
| `NY.GDP.MKTP.PP.KD` | GDP, PPP (constant 2017 international $) |
| `SP.POP.TOTL` | Population, total |
| `SP.POP.1564.TO` | Population ages 15–64 (working age) |

### Download / API Access

- **Bulk download portal:** <https://data.worldbank.org/indicator>
- **World Bank Data API (v2):** `https://api.worldbank.org/v2/country/all/indicator/{INDICATOR_CODE}?format=json`
- **Direct CSV for a single indicator:** accessible via the "Download" button on each indicator page, choosing the "CSV" option which provides a ZIP with a long-format file matching the structure used in this project.

### Column Format

The raw CSV files follow the World Bank standard long format:

| Column | Description |
|--------|-------------|
| `Country Name` | Full country name |
| `Country Code` | ISO 3-letter code |
| `Series Name` | Indicator label (e.g., "GDP (current US$)") |
| `Series Code` | World Bank indicator code |
| `Year` | Year string, often formatted as `"2022 [YR2022]"` |
| `Value` | Numeric value, or `..` for missing |

## Derived / Simulation Output

| File | Description | Availability |
|------|-------------|--------------|
| `migration_sim_results.csv` | Sample output from `run_migration_simulation.py` capturing projected population and GDP per country per year. | ✅ Present in repo (generated) |
