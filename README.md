# GDP and Population Analysis

This project analyzes historical GDP and population data for various countries, providing insights through data cleaning, transformation, exploratory analysis, and an interactive web-based dashboard. It also ships a migration-driven projection engine and helper utilities for exploring how alternative migration flows could shape population and GDP trends over the next several decades.

## Project Overview

### Exploratory Analysis (`gdp_population_analysis.py`)

The exploratory analysis script performs the following key functions:

1.  **Data Loading and Inspection**: It loads country-specific data from a CSV file (`Data_GDP_Pop_by_Country_1960_Countries_only.csv`).
2.  **Data Cleaning and Transformation**:
    *   Missing or placeholder values (e.g., '..', 'N/A') are replaced with `NaN`.
    *   Key columns such as 'GDP', 'PPP_GDP', 'Working_Age_Population', and 'Total_Population' are converted to numeric types.
    *   New metrics are calculated:
        *   GDP per Capita
        *   GDP per Working Age Adult
        *   PPP GDP per Capita
        *   PPP GDP per Working Age Adult
    *   Countries are categorized into 'High Income' (top 20% by GDP per Capita) or 'Other'.
3.  **Exploratory Data Analysis**:
    *   Calculates the correlation between total GDP and GDP per working age adult.
    *   Generates a static scatter plot (using Matplotlib) to visualize 'GDP per Capita' vs. 'GDP per Working Age Adult', color-coded by the 'Income Category'.
    *   Aggregates data by 'Income Category' to show average GDP, average GDP per working age adult, average GDP per capita, and total population for each category.
4.  **Interactive Dashboard**:
    *   Builds a web application using Dash and Plotly Express.
    *   The dashboard features a dropdown to select different metrics for visualization (e.g., 'GDP per Capita', 'PPP GDP per Capita').
    *   Checkboxes allow users to filter countries by 'Income Category' ('High Income', 'Other').
    *   An interactive scatter plot displays the selected metric against 'Total_Population', with points colored by 'Income Category' and country names appearing on hover.

### Migration Projection Toolkit

The migration simulation components allow you to explore multi-decade scenarios by combining baseline population projections with alternative migration assumptions:

* `migration_simulation.MigrationSimulator` orchestrates year-by-year updates of population, working-age cohorts, and GDP per capita. It now validates that input data includes all required columns, prevents division by zero when computing derived metrics, and ignores migration flows that reference unknown countries.
* `migration_simulation.prepare_projection_state` and `migration_simulation.prepare_migration_flows` convert tabular projection and migration CSVs into the dictionary format expected by the simulator.
* `run_migration_simulation.py` ties the helpers and simulator together. It demonstrates how to load repository datasets, configure migration scenarios, and print summarized GDP/population trajectories for the next 50–100 years.

Comprehensive regression tests (`tests/test_migration_helpers.py`) cover the helper functions and simulator safeguards to ensure that scenario runs remain stable as the code evolves.

## How to Run

1.  Ensure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place the data file `Data_GDP_Pop_by_Country_1960_Countries_only.csv` in the same directory as the script, or update the `data_file_path` in `config.json`.
4.  Run the exploratory analysis script:
    ```bash
    python gdp_population_analysis.py
    ```
5.  The Dash application will typically be available at `http://127.0.0.1:8050/` in your web browser.

### Running Migration Scenarios

1.  Ensure the projection (`Data_GDP,PPP_Constant 2021_Working_Age_pop_and_Total.csv`) and migration CSVs are available in the repository.
2.  Execute the scenario driver:
    ```bash
    python run_migration_simulation.py
    ```
3.  Adjust the configuration inside `run_migration_simulation.py` to explore different horizons, migration shocks, or reporting windows.

### Tests

Automated checks validate the helper functions and simulator behaviour:

```bash
pytest -q
```

These components provide a dynamic way to explore and understand both historical relationships and potential futures within the global GDP and population dataset from 1960 onwards.
