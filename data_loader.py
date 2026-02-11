"""Shared data loading and transformation utilities.

This module provides common functions for loading World Bank CSV data and
transforming it into formats suitable for analysis and simulation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.json")


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Optional[Dict[str, Any]]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary, or None if loading fails.
    """
    config_path = Path(config_path)
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Configuration file '%s' not found.", config_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Could not decode JSON from '%s': %s", config_path, e)
        return None


def load_raw_csv(file_path: Path | str) -> Optional[pd.DataFrame]:
    """Load raw CSV data from file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with raw data, or None if loading fails.
    """
    file_path = Path(file_path)
    try:
        data = pd.read_csv(file_path)
        logger.info("Loaded %s with %d rows. Columns: %s", file_path, len(data), data.columns.tolist())
        return data
    except FileNotFoundError:
        logger.error("Data file '%s' not found.", file_path)
        return None
    except pd.errors.EmptyDataError:
        logger.error("Data file '%s' is empty.", file_path)
        return None
    except Exception as e:
        logger.error("Error loading data from '%s': %s", file_path, e)
        return None


def parse_year(year_str: str) -> int:
    """Extract numeric year from World Bank year format.

    World Bank format is like '2022 [YR2022]'.

    Args:
        year_str: Year string in World Bank format.

    Returns:
        Numeric year value.
    """
    if isinstance(year_str, (int, float)):
        return int(year_str)
    # Handle '2022 [YR2022]' format
    return int(str(year_str).split()[0])


def check_data_completeness(
    df: pd.DataFrame,
    value_col: str = "Value",
    country_col: str = "Country Code",
    year_col: str = "Year",
    series_col: str = "Series Name",
    missing_markers: Tuple[str, ...] = ("..", "N/A", ""),
) -> Dict[str, Any]:
    """Analyze data completeness and report missing values.

    Args:
        df: Raw DataFrame in long format.
        value_col: Column containing values.
        country_col: Column containing country codes.
        year_col: Column containing years.
        series_col: Column containing series names.
        missing_markers: Values that indicate missing data.

    Returns:
        Dictionary with completeness statistics:
        - total_rows: Total number of rows
        - missing_count: Number of missing values
        - missing_pct: Percentage of missing values
        - missing_by_series: Missing count per series
        - missing_by_country: Countries with most missing data
        - missing_by_year: Years with most missing data
    """
    df = df.copy()

    # Identify missing values
    is_missing = df[value_col].isin(missing_markers) | df[value_col].isna()

    total_rows = len(df)
    missing_count = is_missing.sum()
    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

    # Missing by series
    missing_by_series = df[is_missing].groupby(series_col).size().to_dict()

    # Top countries with missing data
    missing_by_country = df[is_missing].groupby(country_col).size().sort_values(ascending=False).head(10).to_dict()

    # Missing by year
    missing_by_year = df[is_missing].groupby(year_col).size().sort_values(ascending=False).head(10).to_dict()

    return {
        "total_rows": total_rows,
        "missing_count": int(missing_count),
        "missing_pct": round(missing_pct, 2),
        "missing_by_series": missing_by_series,
        "missing_by_country": missing_by_country,
        "missing_by_year": missing_by_year,
    }


def reshape_long_to_wide(
    df: pd.DataFrame,
    index_cols: List[str],
    columns_col: str,
    values_col: str,
    column_mapping: Optional[Dict[str, str]] = None,
    missing_markers: Tuple[str, ...] = ("..", "N/A", ""),
) -> Optional[pd.DataFrame]:
    """Reshape data from long format to wide format.

    Args:
        df: DataFrame in long format.
        index_cols: Columns to use as index (e.g., ['Country Name', 'Year']).
        columns_col: Column whose values become new column names.
        values_col: Column containing the values to pivot.
        column_mapping: Optional mapping to rename pivoted columns.
        missing_markers: Values to replace with NaN before pivoting.

    Returns:
        Wide-format DataFrame, or None if reshaping fails.
    """
    if df is None:
        return None

    try:
        df = df.copy()

        # Replace missing markers with NaN
        for marker in missing_markers:
            df[values_col] = df[values_col].replace(marker, np.nan)

        # Convert values to numeric
        df[values_col] = pd.to_numeric(df[values_col], errors="coerce")

        # Pivot to wide format
        df_wide = df.pivot_table(
            index=index_cols,
            columns=columns_col,
            values=values_col,
        ).reset_index()

        # Rename columns if mapping provided
        if column_mapping:
            df_wide.rename(columns=column_mapping, inplace=True)

            # Ensure all expected columns exist
            for expected_col in column_mapping.values():
                if expected_col not in df_wide.columns:
                    df_wide[expected_col] = np.nan
                    logger.warning("Added missing expected column: %s", expected_col)

        logger.info("Reshaped data to wide format. Columns: %s", df_wide.columns.tolist())
        return df_wide

    except KeyError as e:
        logger.error("Error during pivoting: Missing key %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error during data reshaping: %s", e)
        return None


def calculate_derived_metrics(
    df: pd.DataFrame,
    gdp_col: str = "GDP",
    ppp_gdp_col: str = "PPP_GDP",
    total_pop_col: str = "Total_Population",
    working_age_col: str = "Working_Age_Population",
) -> pd.DataFrame:
    """Calculate derived economic metrics.

    Safely handles division by zero by returning NaN for those cases.

    Args:
        df: DataFrame with base economic columns.
        gdp_col: Column name for GDP values.
        ppp_gdp_col: Column name for PPP GDP values.
        total_pop_col: Column name for total population.
        working_age_col: Column name for working age population.

    Returns:
        DataFrame with added derived columns.
    """
    df = df.copy()

    # Ensure numeric types
    for col in [gdp_col, ppp_gdp_col, total_pop_col, working_age_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate metrics with safe division (returns NaN for division by zero)
    if gdp_col in df.columns and total_pop_col in df.columns:
        df["GDP per Capita"] = df[gdp_col] / df[total_pop_col].replace(0, np.nan)
    else:
        df["GDP per Capita"] = np.nan

    if gdp_col in df.columns and working_age_col in df.columns:
        df["GDP per Working Age Adult"] = df[gdp_col] / df[working_age_col].replace(0, np.nan)
    else:
        df["GDP per Working Age Adult"] = np.nan

    if ppp_gdp_col in df.columns and total_pop_col in df.columns:
        df["PPP GDP per Capita"] = df[ppp_gdp_col] / df[total_pop_col].replace(0, np.nan)
    else:
        df["PPP GDP per Capita"] = np.nan

    if ppp_gdp_col in df.columns and working_age_col in df.columns:
        df["PPP GDP per Working Age Adult"] = df[ppp_gdp_col] / df[working_age_col].replace(0, np.nan)
    else:
        df["PPP GDP per Working Age Adult"] = np.nan

    return df


def categorize_income(
    df: pd.DataFrame,
    gdp_per_capita_col: str = "GDP per Capita",
    percentile_threshold: float = 80,
) -> pd.DataFrame:
    """Categorize countries by income level.

    Args:
        df: DataFrame with GDP per capita column.
        gdp_per_capita_col: Column name for GDP per capita.
        percentile_threshold: Percentile threshold for high income classification.

    Returns:
        DataFrame with added 'Income Category' column.
    """
    df = df.copy()

    if gdp_per_capita_col in df.columns and not df[gdp_per_capita_col].isna().all():
        threshold_value = np.nanpercentile(df[gdp_per_capita_col].dropna(), percentile_threshold)
        df["Income Category"] = np.where(df[gdp_per_capita_col] >= threshold_value, "High Income", "Other")
    else:
        logger.warning("Cannot categorize income: '%s' missing or all NaN", gdp_per_capita_col)
        df["Income Category"] = "Other"

    return df


def load_and_transform(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    data_path: Optional[Path | str] = None,
) -> Optional[pd.DataFrame]:
    """Load and fully transform data using configuration.

    This is a convenience function that combines loading, reshaping,
    calculating derived metrics, and income categorization.

    Args:
        config_path: Path to configuration file.
        data_path: Optional override for data file path.

    Returns:
        Fully transformed DataFrame, or None if processing fails.
    """
    config = load_config(config_path)
    if config is None:
        return None

    file_path = data_path or config.get("data_file_path")
    if not file_path:
        logger.error("No data file path specified")
        return None

    raw_data = load_raw_csv(file_path)
    if raw_data is None:
        return None

    # Reshape to wide format
    pivot_config = {
        "index_cols": config.get("pivot_index_columns", ["Country Name", "Year", "Country Code"]),
        "columns_col": config.get("pivot_columns_column", "Series Name"),
        "values_col": config.get("pivot_values_column", "Value"),
        "column_mapping": config.get("series_name_mapping"),
    }

    wide_data = reshape_long_to_wide(raw_data, **pivot_config)
    if wide_data is None:
        return None

    # Calculate derived metrics
    wide_data = calculate_derived_metrics(wide_data)

    # Categorize income
    threshold = config.get("high_income_percentile_threshold", 80)
    wide_data = categorize_income(wide_data, percentile_threshold=threshold)

    return wide_data


def prepare_for_simulation(
    df: pd.DataFrame,
    country_col: str = "Country Name",
    year_col: str = "Year",
    gdp_col: str = "GDP",
    pop_col: str = "Total_Population",
) -> pd.DataFrame:
    """Prepare data for MigrationSimulator.

    Renames columns and filters to match simulator expectations.

    Args:
        df: Transformed DataFrame.
        country_col: Source column name for country.
        year_col: Source column name for year.
        gdp_col: Source column name for GDP.
        pop_col: Source column name for population.

    Returns:
        DataFrame with columns expected by MigrationSimulator.
    """
    df = df.copy()

    # Rename to match simulator expectations
    rename_map = {}
    if country_col != "Country" and country_col in df.columns:
        rename_map[country_col] = "Country"
    if year_col != "Year" and year_col in df.columns:
        rename_map[year_col] = "Year"
    if gdp_col != "GDP" and gdp_col in df.columns:
        rename_map[gdp_col] = "GDP"
    if pop_col != "Total_Population" and pop_col in df.columns:
        rename_map[pop_col] = "Total_Population"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Parse year if needed
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(parse_year)

    # Filter out rows with missing required data
    required = ["Country", "Year", "GDP", "Total_Population"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Filter out zero/negative population
    if "Total_Population" in df.columns:
        df = df[df["Total_Population"] > 0]

    return df


__all__ = [
    "load_config",
    "load_raw_csv",
    "parse_year",
    "check_data_completeness",
    "reshape_long_to_wide",
    "calculate_derived_metrics",
    "categorize_income",
    "load_and_transform",
    "prepare_for_simulation",
]
