import pandas as pd
import pytest

from migration_simulation import (
    MigrationSimulator,
    dataframe_to_projection_dict,
    migration_flows_from_dataframe,
    validate_base_dataframe,
)


def test_dataframe_to_projection_dict_basic():
    df = pd.DataFrame(
        {
            "Country": ["A", "A", "B"],
            "Year": [2020, 2021, 2021],
            "Growth": [0.01, 0.02, 0.03],
        }
    )

    projection = dataframe_to_projection_dict(
        df,
        country_col="Country",
        year_col="Year",
        value_col="Growth",
    )

    assert projection[("A", 2020)] == pytest.approx(0.01)
    assert projection[("A", 2021)] == pytest.approx(0.02)
    assert projection[("B", 2021)] == pytest.approx(0.03)


def test_migration_flows_from_dataframe_skips_missing_and_requires_non_negative():
    df = pd.DataFrame(
        {
            "Origin": ["A", "A", None],
            "Destination": ["B", "C", "D"],
            "Year": [2025, 2025, 2025],
            "Migrants": [100, -50, 200],
        }
    )

    flows = migration_flows_from_dataframe(
        df.drop(index=1),
        origin_col="Origin",
        destination_col="Destination",
        year_col="Year",
        flow_col="Migrants",
    )

    assert flows == {2025: [("A", "B", 100.0)]}

    with pytest.raises(ValueError):
        migration_flows_from_dataframe(
            df,
            origin_col="Origin",
            destination_col="Destination",
            year_col="Year",
            flow_col="Migrants",
        )


def test_validate_base_dataframe_and_simulator_handles_unknown_destinations():
    base_df = pd.DataFrame(
        {
            "Country": ["A", "B"],
            "Year": [2020, 2020],
            "Total_Population": [1_000, 500],
            "GDP": [1000, 600],
        }
    )

    validate_base_dataframe(base_df, MigrationSimulator.REQUIRED_COLUMNS)

    pop_proj = {("A", 2021): 0.0, ("B", 2021): 0.0}
    gdp_proj = {("A", 2021): 0.0, ("B", 2021): 0.0}
    flows = {2021: [("A", "B", 100), ("B", "C", 50)]}

    sim = MigrationSimulator(base_df, pop_proj, gdp_proj, flows, init_prod=0.5, prod_step=0.5)
    sim.step()

    # 100 people move from A to B; flow to missing destination is skipped
    assert sim.state["A"]["pop"] == pytest.approx(900)
    assert sim.state["B"]["pop"] == pytest.approx(600)

    with pytest.raises(ValueError):
        invalid_base = base_df.copy()
        invalid_base.loc[0, "Total_Population"] = 0
        MigrationSimulator(invalid_base, pop_proj, gdp_proj, {})
