"""Example entry point for running the :class:`MigrationSimulator`.

The helper utilities provided in :mod:`migration_simulation` make it easy to
transform CSV projections into the dictionaries expected by the simulator.  This
script demonstrates how those utilities can be composed to build a simple
scenario using the historical data already present in the repository.  Replace
the example data loading logic with your own projection sources when running a
full analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from migration_simulation import (
    MigrationSimulator,
    MigrationFlowsDict,
    ProjectionDict,
    dataframe_to_projection_dict,
    migration_flows_from_dataframe,
    validate_base_dataframe,
)

BASE_DATA_PATH = Path("Data_GDP_Pop_by_Country_1960_Countries_only.csv")
OUTPUT_PATH = Path("simulation_results_example.csv")


def build_example_inputs(
    base_csv: Path,
) -> Tuple[pd.DataFrame, ProjectionDict, ProjectionDict, MigrationFlowsDict]:
    """Construct a minimal set of simulator inputs from the repository dataset."""

    base_df = pd.read_csv(base_csv)
    latest_year = int(base_df["Year"].max())
    snapshot = base_df[base_df["Year"] == latest_year].copy()
    validate_base_dataframe(snapshot, MigrationSimulator.REQUIRED_COLUMNS)

    history = base_df.sort_values(["Country", "Year"]).copy()
    history["gdp_pc"] = history["GDP"] / history["Total_Population"]
    history["pop_growth"] = history.groupby("Country")["Total_Population"].pct_change()
    history["gdp_pc_growth"] = history.groupby("Country")["gdp_pc"].pct_change()

    pop_growth = dataframe_to_projection_dict(
        history.dropna(subset=["pop_growth"]),
        country_col="Country",
        year_col="Year",
        value_col="pop_growth",
    )
    gdp_growth = dataframe_to_projection_dict(
        history.dropna(subset=["gdp_pc_growth"]),
        country_col="Country",
        year_col="Year",
        value_col="gdp_pc_growth",
    )

    migration_df = pd.DataFrame(
        [
            {
                "Year": latest_year + 1,
                "Origin": "United States",
                "Destination": "Canada",
                "Migrants": 50_000,
            },
            {
                "Year": latest_year + 1,
                "Origin": "Canada",
                "Destination": "United States",
                "Migrants": 30_000,
            },
        ]
    )
    migration_flows = migration_flows_from_dataframe(
        migration_df,
        origin_col="Origin",
        destination_col="Destination",
        year_col="Year",
        flow_col="Migrants",
    )

    return snapshot, pop_growth, gdp_growth, migration_flows


def main() -> None:
    base_data, pop_growth, gdp_growth, migration_flows = build_example_inputs(BASE_DATA_PATH)

    simulator = MigrationSimulator(
        base_data,
        pop_growth=pop_growth,
        gdp_proj=gdp_growth,
        migration_flows=migration_flows,
        init_prod=0.5,
        prod_step=0.1,
    )
    end_year = simulator.year + 10
    results = simulator.run(end_year=end_year)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved example simulation covering up to {end_year} to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
