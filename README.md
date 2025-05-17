# GDP and Population Analysis

This project analyzes historical GDP and population data for various countries, providing insights through data cleaning, transformation, exploratory analysis, and an interactive web-based dashboard.

## Project Overview

The Python script (`code.py`) performs the following key functions:

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

## How to Run

1.  Ensure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install pandas numpy matplotlib dash plotly
    ```
3.  Place the data file `Data_GDP_Pop_by_Country_1960_Countries_only.csv` in the specified path within `code.py` or update the `file_path` variable in the script.
4.  Run the script:
    ```bash
    python code.py
    ```
5.  The Dash application will typically be available at `http://127.0.0.1:8050/` in your web browser.

This tool provides a dynamic way to explore and understand relationships within the global GDP and population dataset from 1960 onwards.
