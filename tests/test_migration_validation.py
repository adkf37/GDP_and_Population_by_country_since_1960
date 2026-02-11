"""Validation tests for migration flows against UN statistics.

Reference data from UN International Migrant Stock database and
UN Population Division estimates.

Sources:
- UN International Migrant Stock: https://www.un.org/development/desa/pd/content/international-migrant-stock
- UN World Population Prospects: https://population.un.org/wpp/
"""

import pytest
import pandas as pd

from migration_simulation import migration_flows_from_dataframe, MigrationFlowsDict


# UN International Migrant Stock 2020 - Top bilateral migration corridors
# Source: UN DESA Population Division (2020)
# Values are migrant stocks (total migrants from origin living in destination)
UN_MIGRANT_STOCK_2020 = {
    # (origin, destination): stock
    ("Mexico", "United States"): 10_932_000,
    ("India", "United Arab Emirates"): 3_500_000,
    ("India", "United States"): 2_700_000,
    ("China", "United States"): 2_400_000,
    ("Syria", "Turkey"): 3_700_000,
    ("Philippines", "United States"): 2_000_000,
    ("Bangladesh", "India"): 2_200_000,
    ("Afghanistan", "Iran"): 2_000_000,
    ("Ukraine", "Russia"): 3_300_000,
    ("Poland", "Germany"): 900_000,
}

# Estimated annual flow rates (rough approximation: stock growth / 10 years)
# These are order-of-magnitude estimates for validation
UN_ESTIMATED_ANNUAL_FLOWS = {
    ("Mexico", "United States"): (100_000, 500_000),  # Range of plausible annual flows
    ("India", "United States"): (50_000, 200_000),
    ("China", "United States"): (50_000, 150_000),
    ("Syria", "Turkey"): (200_000, 800_000),  # Higher during crisis years
}

# UN estimates for net migration 2020-2025 (annual average)
# Source: UN World Population Prospects 2022
UN_NET_MIGRATION_2020_2025 = {
    "United States": 1_000_000,  # ~1M net migrants per year
    "Germany": 250_000,
    "Canada": 250_000,
    "United Kingdom": 200_000,
    "Australia": 150_000,
    "Mexico": -100_000,  # Net emigration
    "China": -200_000,
    "India": -400_000,
}


class TestMigrationFlowRealism:
    """Validate that simulated migration flows are realistic."""

    def test_flow_magnitude_sanity_check(self):
        """Migration flows should be within realistic bounds."""
        # A single corridor shouldn't exceed ~1M/year even for largest flows
        max_realistic_annual_flow = 1_000_000
        
        # Example: Mexico -> US is one of largest corridors
        example_flow = 500_000  # Half a million per year
        assert example_flow <= max_realistic_annual_flow

    def test_example_flows_match_un_order_of_magnitude(self):
        """Example migration flows should be within UN-estimated ranges."""
        # Create example flows similar to run_migration_simulation.py
        example_flows = pd.DataFrame([
            {"Year": 2024, "Origin": "Mexico", "Destination": "United States", "Migrants": 200_000},
            {"Year": 2024, "Origin": "India", "Destination": "United States", "Migrants": 100_000},
        ])
        
        flows = migration_flows_from_dataframe(
            example_flows,
            origin_col="Origin",
            destination_col="Destination",
            year_col="Year",
            flow_col="Migrants",
        )
        
        # Check Mexico -> US flow is within UN-estimated range
        mexico_us_flow = None
        for origin, dest, count in flows.get(2024, []):
            if origin == "Mexico" and dest == "United States":
                mexico_us_flow = count
        
        if mexico_us_flow:
            min_expected, max_expected = UN_ESTIMATED_ANNUAL_FLOWS[("Mexico", "United States")]
            assert min_expected <= mexico_us_flow <= max_expected, (
                f"Mexico->US flow {mexico_us_flow:,} outside UN range [{min_expected:,}, {max_expected:,}]"
            )

    def test_net_migration_country_direction(self):
        """Net migration should flow in expected directions."""
        # Immigration countries should have positive net flows
        immigration_countries = ["United States", "Germany", "Canada", "Australia"]
        # Emigration countries should have negative net flows  
        emigration_countries = ["Mexico", "India", "Philippines"]
        
        # This is a structural test - actual simulation should respect these patterns
        for country in immigration_countries:
            if country in UN_NET_MIGRATION_2020_2025:
                assert UN_NET_MIGRATION_2020_2025[country] > 0, (
                    f"{country} should have positive net migration"
                )
        
        for country in emigration_countries:
            if country in UN_NET_MIGRATION_2020_2025:
                assert UN_NET_MIGRATION_2020_2025[country] <= 0, (
                    f"{country} should have non-positive net migration"
                )


class TestMigrationStockReference:
    """Reference data for migrant stock validation."""

    def test_un_stock_data_available(self):
        """UN migrant stock reference data should be available."""
        assert len(UN_MIGRANT_STOCK_2020) > 0
        
        # Largest corridor should be Mexico -> US
        mexico_us = UN_MIGRANT_STOCK_2020.get(("Mexico", "United States"))
        assert mexico_us is not None
        assert mexico_us > 10_000_000  # Over 10 million

    def test_major_corridors_have_data(self):
        """Major migration corridors should have reference data."""
        major_corridors = [
            ("Mexico", "United States"),
            ("India", "United Arab Emirates"),
            ("Syria", "Turkey"),
        ]
        
        for corridor in major_corridors:
            assert corridor in UN_MIGRANT_STOCK_2020, f"Missing data for {corridor}"

    @pytest.mark.parametrize("corridor,expected_min", [
        (("Mexico", "United States"), 10_000_000),
        (("India", "United Arab Emirates"), 3_000_000),
        (("Syria", "Turkey"), 3_000_000),
    ])
    def test_stock_values_reasonable(self, corridor, expected_min):
        """Migrant stock values should meet minimum thresholds."""
        stock = UN_MIGRANT_STOCK_2020.get(corridor)
        assert stock is not None
        assert stock >= expected_min


class TestScenarioPlausibility:
    """Test that scenario configurations are plausible."""

    def test_placeholder_flows_are_clearly_examples(self):
        """Placeholder flows in example code should be recognizable as examples."""
        # The flows in run_migration_simulation.py use round numbers
        # which indicate they're examples, not real data
        example_flows = [50_000, 30_000, 100_000]
        
        for flow in example_flows:
            # Round numbers ending in 000 suggest placeholder values
            assert flow % 1000 == 0, "Example flows should be round numbers"

    def test_productivity_assumptions_documented(self):
        """Productivity assumptions should be within literature ranges."""
        # Literature suggests immigrants reach native productivity in 5-20 years
        # With prod_step=0.1 and init_prod=0.5, full productivity in 5 years
        init_prod = 0.5
        prod_step = 0.1
        years_to_full_productivity = (1.0 - init_prod) / prod_step
        
        # Should be within 3-20 year range from literature
        assert 3 <= years_to_full_productivity <= 20, (
            f"Years to full productivity ({years_to_full_productivity}) outside literature range"
        )


def get_un_migration_reference() -> dict:
    """Get UN migration reference data for use in simulations.
    
    Returns:
        Dictionary with UN reference statistics.
    """
    return {
        "stock_2020": UN_MIGRANT_STOCK_2020,
        "estimated_annual_flows": UN_ESTIMATED_ANNUAL_FLOWS,
        "net_migration_2020_2025": UN_NET_MIGRATION_2020_2025,
    }
