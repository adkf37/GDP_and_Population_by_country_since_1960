"""Tests for the shared data_loader module.

Includes edge case tests for reshape_data and integration tests
that load real CSV data and run through the full transformation pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    parse_year,
    check_data_completeness,
    reshape_long_to_wide,
    calculate_derived_metrics,
    categorize_income,
    load_and_transform,
    prepare_for_simulation,
)

DATA_FILE = Path(__file__).parent.parent / "Data_GDP_Pop_by_Country_1960_Countries_only.csv"
CONFIG_FILE = Path(__file__).parent.parent / "config.json"


class TestParseYear:
    """Tests for year parsing utility."""

    def test_parse_world_bank_format(self):
        """Parse '2022 [YR2022]' format correctly."""
        assert parse_year("2022 [YR2022]") == 2022
        assert parse_year("1960 [YR1960]") == 1960

    def test_parse_integer(self):
        """Handle integer input."""
        assert parse_year(2022) == 2022

    def test_parse_float(self):
        """Handle float input."""
        assert parse_year(2022.0) == 2022

    def test_parse_simple_string(self):
        """Handle simple string year."""
        assert parse_year("2022") == 2022


class TestReshapeLongToWide:
    """Edge case tests for reshape_long_to_wide."""

    def test_handles_none_input(self):
        """Return None for None input."""
        result = reshape_long_to_wide(
            None,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
        )
        assert result is None

    def test_handles_missing_markers(self):
        """Replace missing value markers with NaN."""
        df = pd.DataFrame({
            "Country": ["A", "A", "B", "B"],
            "Year": [2020, 2020, 2020, 2020],
            "Series": ["GDP", "Pop", "GDP", "Pop"],
            "Value": ["1000", "..", "2000", "500"],  # B has valid Pop
        })
        
        result = reshape_long_to_wide(
            df,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
        )
        
        assert result is not None
        # A's Pop was '..' so should be NaN
        assert pd.isna(result.loc[result["Country"] == "A", "Pop"].iloc[0])
        # B's Pop was valid
        assert result.loc[result["Country"] == "B", "Pop"].iloc[0] == 500.0
        assert result.loc[result["Country"] == "A", "GDP"].iloc[0] == 1000.0

    def test_handles_duplicate_entries(self):
        """Aggregate duplicates using pivot_table default (mean)."""
        df = pd.DataFrame({
            "Country": ["A", "A", "A"],
            "Year": [2020, 2020, 2020],
            "Series": ["GDP", "GDP", "GDP"],
            "Value": [1000, 2000, 3000],
        })
        
        result = reshape_long_to_wide(
            df,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
        )
        
        assert result is not None
        # pivot_table takes mean by default for duplicates
        assert result.loc[0, "GDP"] == pytest.approx(2000.0)

    def test_handles_missing_series(self):
        """Create columns with NaN for missing series."""
        df = pd.DataFrame({
            "Country": ["A", "B"],
            "Year": [2020, 2020],
            "Series": ["GDP", "Pop"],  # A only has GDP, B only has Pop
            "Value": [1000, 500],
        })
        
        result = reshape_long_to_wide(
            df,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
        )
        
        assert result is not None
        # A should have NaN for Pop
        a_row = result[result["Country"] == "A"]
        assert pd.isna(a_row["Pop"].iloc[0])
        assert a_row["GDP"].iloc[0] == 1000.0

    def test_column_mapping_adds_missing_columns(self):
        """Column mapping should add missing expected columns."""
        df = pd.DataFrame({
            "Country": ["A"],
            "Year": [2020],
            "Series": ["GDP"],
            "Value": [1000],
        })
        
        mapping = {"GDP": "gdp_value", "Pop": "pop_value"}
        
        result = reshape_long_to_wide(
            df,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
            column_mapping=mapping,
        )
        
        assert result is not None
        assert "gdp_value" in result.columns
        assert "pop_value" in result.columns  # Should be added with NaN
        assert pd.isna(result["pop_value"].iloc[0])

    def test_handles_malformed_numeric_values(self):
        """Convert non-numeric values to NaN (rows with all NaN are dropped by pivot)."""
        df = pd.DataFrame({
            "Country": ["A", "A", "B", "B"],
            "Year": [2020, 2020, 2020, 2020],
            "Series": ["GDP", "Pop", "GDP", "Pop"],
            "Value": ["not_a_number", "abc123", "1000", "500"],  # B has valid values
        })
        
        result = reshape_long_to_wide(
            df,
            index_cols=["Country", "Year"],
            columns_col="Series",
            values_col="Value",
        )
        
        assert result is not None
        # A's values were all malformed/NaN, so row is dropped by pivot_table
        assert len(result[result["Country"] == "A"]) == 0
        # B's values were valid and should be present
        b_row = result[result["Country"] == "B"]
        assert len(b_row) == 1
        assert b_row["GDP"].iloc[0] == 1000.0
        assert b_row["Pop"].iloc[0] == 500.0


class TestCalculateDerivedMetrics:
    """Tests for derived metric calculations."""

    def test_handles_zero_population(self):
        """Division by zero population returns NaN, not infinity."""
        df = pd.DataFrame({
            "GDP": [1000, 2000],
            "PPP_GDP": [1100, 2100],
            "Total_Population": [0, 100],
            "Working_Age_Population": [0, 50],
        })
        
        result = calculate_derived_metrics(df)
        
        # Zero population should result in NaN, not inf
        assert pd.isna(result.loc[0, "GDP per Capita"])
        assert pd.isna(result.loc[0, "GDP per Working Age Adult"])
        assert pd.isna(result.loc[0, "PPP GDP per Capita"])
        
        # Non-zero population should calculate correctly
        assert result.loc[1, "GDP per Capita"] == pytest.approx(20.0)
        assert result.loc[1, "GDP per Working Age Adult"] == pytest.approx(40.0)

    def test_handles_missing_columns(self):
        """Return NaN columns when source columns are missing."""
        df = pd.DataFrame({
            "GDP": [1000],
            # Missing PPP_GDP, Working_Age_Population
            "Total_Population": [100],
        })
        
        result = calculate_derived_metrics(df)
        
        assert result.loc[0, "GDP per Capita"] == pytest.approx(10.0)
        assert pd.isna(result.loc[0, "PPP GDP per Capita"])
        assert pd.isna(result.loc[0, "GDP per Working Age Adult"])


class TestDataCompleteness:
    """Tests for data completeness checking."""

    def test_reports_missing_values(self):
        """Correctly count and categorize missing values."""
        df = pd.DataFrame({
            "Country Code": ["USA", "USA", "CHN", "CHN"],
            "Year": ["2020", "2021", "2020", "2021"],
            "Series Name": ["GDP", "GDP", "GDP", "GDP"],
            "Value": ["1000", "..", "N/A", "2000"],
        })
        
        result = check_data_completeness(df)
        
        assert result["total_rows"] == 4
        assert result["missing_count"] == 2
        assert result["missing_pct"] == 50.0
        assert result["missing_by_series"]["GDP"] == 2


class TestIntegration:
    """Integration tests using real data."""

    @pytest.fixture
    def transformed_data(self):
        """Load and transform real data."""
        if not DATA_FILE.exists() or not CONFIG_FILE.exists():
            pytest.skip("Data or config file not found")
        return load_and_transform(CONFIG_FILE)

    def test_load_and_transform_returns_dataframe(self, transformed_data):
        """Full pipeline produces a valid DataFrame."""
        assert transformed_data is not None
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) > 0

    def test_transformed_data_has_expected_columns(self, transformed_data):
        """Transformed data has all derived columns."""
        expected_cols = [
            "Country Name",
            "Year",
            "GDP",
            "Total_Population",
            "GDP per Capita",
            "Income Category",
        ]
        for col in expected_cols:
            assert col in transformed_data.columns, f"Missing column: {col}"

    def test_prepare_for_simulation(self, transformed_data):
        """Data can be prepared for MigrationSimulator."""
        sim_data = prepare_for_simulation(transformed_data)
        
        assert "Country" in sim_data.columns
        assert "Year" in sim_data.columns
        assert "GDP" in sim_data.columns
        assert "Total_Population" in sim_data.columns
        
        # Year should be numeric
        assert sim_data["Year"].dtype in [int, np.int64, np.int32]
        
        # No zero populations
        assert (sim_data["Total_Population"] > 0).all()

    def test_end_to_end_simulation_setup(self, transformed_data):
        """Data can be loaded and prepared for a full simulation run."""
        from migration_simulation import MigrationSimulator, validate_base_dataframe
        
        sim_data = prepare_for_simulation(transformed_data)
        latest_year = int(sim_data["Year"].max())
        snapshot = sim_data[sim_data["Year"] == latest_year].copy()
        
        # Should pass validation
        validate_base_dataframe(snapshot, MigrationSimulator.REQUIRED_COLUMNS)
        
        # Should have multiple countries
        assert len(snapshot) > 10
        
        # Should be able to create simulator (basic smoke test)
        pop_proj = {(snapshot.iloc[0]["Country"], latest_year + 1): 0.01}
        gdp_proj = {(snapshot.iloc[0]["Country"], latest_year + 1): 0.02}
        
        sim = MigrationSimulator(
            snapshot,
            pop_proj=pop_proj,
            gdp_proj=gdp_proj,
            migration_flows={},
        )
        assert sim.year == latest_year
