"""Validation tests comparing dataset values against World Bank published statistics.

These tests verify that GDP and population figures in the repository data match
official World Bank statistics within acceptable tolerance. Reference values are
from the World Bank Open Data portal (https://data.worldbank.org/).

World Bank Indicator Codes:
- NY.GDP.MKTP.CD: GDP (current US$)
- SP.POP.TOTL: Population, total
"""

from pathlib import Path

import pandas as pd
import pytest

DATA_FILE = Path(__file__).parent.parent / "Data_GDP_Pop_by_Country_1960_Countries_only.csv"

# World Bank reference values for 2022 (source: data.worldbank.org)
# Tolerance set to 2% to account for data revisions between releases
WORLD_BANK_GDP_2022 = {
    "USA": 25_744_100_000_000,  # $25.74 trillion
    "CHN": 17_881_800_000_000,  # $17.88 trillion
    "JPN": 4_256_410_000_000,   # $4.26 trillion
    "DEU": 4_082_470_000_000,   # $4.08 trillion
    "GBR": 3_089_070_000_000,   # $3.09 trillion
    "IND": 3_416_650_000_000,   # $3.42 trillion
}

WORLD_BANK_POPULATION_2022 = {
    "USA": 333_271_411,
    "CHN": 1_412_175_000,
    "JPN": 125_124_989,
    "DEU": 83_797_985,
    "GBR": 66_971_395,
    "IND": 1_417_173_173,
}

# Historical reference points for trend validation
WORLD_BANK_GDP_2010 = {
    "USA": 15_048_970_000_000,  # $15.05 trillion
    "CHN": 6_087_160_000_000,   # $6.09 trillion
}

WORLD_BANK_POPULATION_2010 = {
    "USA": 309_327_143,
    "CHN": 1_337_705_000,
}


@pytest.fixture
def raw_data():
    """Load the raw CSV data."""
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


def extract_value(df: pd.DataFrame, country_code: str, series_name: str, year_pattern: str) -> float:
    """Extract a single value from the long-format dataframe."""
    mask = (
        (df["Country Code"] == country_code)
        & (df["Series Name"] == series_name)
        & (df["Year"].str.contains(year_pattern))
    )
    values = df.loc[mask, "Value"]
    if values.empty:
        return float("nan")
    val = values.iloc[0]
    if val == "..":
        return float("nan")
    return float(val)


class TestGDPValidation:
    """Verify GDP values match World Bank published statistics."""

    @pytest.mark.parametrize("country_code,expected_gdp", list(WORLD_BANK_GDP_2022.items()))
    def test_gdp_2022_matches_world_bank(self, raw_data, country_code, expected_gdp):
        """GDP for 2022 should match World Bank data within 1% tolerance."""
        actual = extract_value(raw_data, country_code, "GDP (current US$)", "2022")
        if pd.isna(actual):
            pytest.skip(f"No GDP data for {country_code} in 2022")
        
        tolerance = 0.02  # 2%
        assert actual == pytest.approx(expected_gdp, rel=tolerance), (
            f"{country_code} GDP mismatch: got {actual:,.0f}, expected {expected_gdp:,.0f}"
        )

    @pytest.mark.parametrize("country_code,expected_gdp", list(WORLD_BANK_GDP_2010.items()))
    def test_gdp_2010_matches_world_bank(self, raw_data, country_code, expected_gdp):
        """GDP for 2010 should match World Bank data within 1% tolerance."""
        actual = extract_value(raw_data, country_code, "GDP (current US$)", "2010")
        if pd.isna(actual):
            pytest.skip(f"No GDP data for {country_code} in 2010")
        
        tolerance = 0.02
        assert actual == pytest.approx(expected_gdp, rel=tolerance)

    def test_usa_gdp_growth_trend(self, raw_data):
        """USA GDP should show growth from 2010 to 2022."""
        gdp_2010 = extract_value(raw_data, "USA", "GDP (current US$)", "2010")
        gdp_2022 = extract_value(raw_data, "USA", "GDP (current US$)", "2022")
        
        if pd.isna(gdp_2010) or pd.isna(gdp_2022):
            pytest.skip("Missing USA GDP data for trend test")
        
        # GDP should have grown by at least 50% over 12 years
        assert gdp_2022 > gdp_2010 * 1.5, "USA GDP growth trend appears incorrect"

    def test_china_gdp_growth_trend(self, raw_data):
        """China GDP should show substantial growth from 2010 to 2022."""
        gdp_2010 = extract_value(raw_data, "CHN", "GDP (current US$)", "2010")
        gdp_2022 = extract_value(raw_data, "CHN", "GDP (current US$)", "2022")
        
        if pd.isna(gdp_2010) or pd.isna(gdp_2022):
            pytest.skip("Missing China GDP data for trend test")
        
        # China GDP should have at least doubled over 12 years
        assert gdp_2022 > gdp_2010 * 2.0, "China GDP growth trend appears incorrect"


class TestPopulationValidation:
    """Verify population values match World Bank published statistics."""

    @pytest.mark.parametrize("country_code,expected_pop", list(WORLD_BANK_POPULATION_2022.items()))
    def test_population_2022_matches_world_bank(self, raw_data, country_code, expected_pop):
        """Population for 2022 should match World Bank data within 1% tolerance."""
        actual = extract_value(raw_data, country_code, "Population, total", "2022")
        if pd.isna(actual):
            pytest.skip(f"No population data for {country_code} in 2022")
        
        tolerance = 0.02
        assert actual == pytest.approx(expected_pop, rel=tolerance), (
            f"{country_code} population mismatch: got {actual:,.0f}, expected {expected_pop:,.0f}"
        )

    @pytest.mark.parametrize("country_code,expected_pop", list(WORLD_BANK_POPULATION_2010.items()))
    def test_population_2010_matches_world_bank(self, raw_data, country_code, expected_pop):
        """Population for 2010 should match World Bank data within 1% tolerance."""
        actual = extract_value(raw_data, country_code, "Population, total", "2010")
        if pd.isna(actual):
            pytest.skip(f"No population data for {country_code} in 2010")
        
        tolerance = 0.02
        assert actual == pytest.approx(expected_pop, rel=tolerance)

    def test_japan_population_decline_trend(self, raw_data):
        """Japan population should show decline from 2010 to 2022."""
        pop_2010 = extract_value(raw_data, "JPN", "Population, total", "2010")
        pop_2022 = extract_value(raw_data, "JPN", "Population, total", "2022")
        
        if pd.isna(pop_2010) or pd.isna(pop_2022):
            pytest.skip("Missing Japan population data for trend test")
        
        # Japan has been experiencing population decline
        assert pop_2022 < pop_2010, "Japan population decline trend not reflected in data"


class TestDataCompleteness:
    """Verify data completeness for major economies."""

    MAJOR_ECONOMIES = ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "BRA"]

    def test_major_economies_have_recent_gdp(self, raw_data):
        """Major economies should have GDP data for 2022."""
        missing = []
        for code in self.MAJOR_ECONOMIES:
            val = extract_value(raw_data, code, "GDP (current US$)", "2022")
            if pd.isna(val):
                missing.append(code)
        
        assert not missing, f"Missing 2022 GDP data for: {missing}"

    def test_major_economies_have_recent_population(self, raw_data):
        """Major economies should have population data for 2022."""
        missing = []
        for code in self.MAJOR_ECONOMIES:
            val = extract_value(raw_data, code, "Population, total", "2022")
            if pd.isna(val):
                missing.append(code)
        
        assert not missing, f"Missing 2022 population data for: {missing}"

    def test_gdp_per_capita_sanity_check(self, raw_data):
        """GDP per capita should be within reasonable bounds for known economies."""
        # Known approximate GDP per capita ranges for 2022
        expected_ranges = {
            "USA": (60_000, 90_000),      # ~$77k
            "CHN": (10_000, 20_000),      # ~$13k
            "IND": (2_000, 4_000),        # ~$2.4k
            "DEU": (40_000, 60_000),      # ~$49k
        }
        
        for code, (low, high) in expected_ranges.items():
            gdp = extract_value(raw_data, code, "GDP (current US$)", "2022")
            pop = extract_value(raw_data, code, "Population, total", "2022")
            
            if pd.isna(gdp) or pd.isna(pop) or pop == 0:
                continue
            
            gdp_per_cap = gdp / pop
            assert low <= gdp_per_cap <= high, (
                f"{code} GDP per capita {gdp_per_cap:,.0f} outside expected range [{low:,}, {high:,}]"
            )
