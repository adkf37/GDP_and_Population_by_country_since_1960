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
from typing import Any, Dict, Tuple

import pandas as pd

from data_loader import (
    load_config,
    load_and_transform,
    prepare_for_simulation,
    parse_year,
)
from migration_simulation import (
    MigrationSimulator,
    MigrationFlowsDict,
    ProjectionDict,
    dataframe_to_projection_dict,
    migration_flows_from_dataframe,
    validate_base_dataframe,
)

CONFIG_PATH = Path("config.json")
OUTPUT_PATH = Path("simulation_results_example.csv")


def build_example_inputs(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, ProjectionDict, ProjectionDict, MigrationFlowsDict]:
    """Construct a minimal set of simulator inputs from the repository dataset.
    
    Uses the shared data_loader module to properly transform World Bank
    long-format CSV to the wide format expected by MigrationSimulator.
    """
    # Load and transform data using shared utilities
    transformed = load_and_transform(CONFIG_PATH)
    if transformed is None:
        raise RuntimeError("Failed to load and transform data")
    
    # Prepare for simulation (renames columns, parses years, filters)
    sim_data = prepare_for_simulation(transformed)
    
    # Get latest year snapshot for base data
    latest_year = int(sim_data["Year"].max())
    snapshot = sim_data[sim_data["Year"] == latest_year].copy()
    validate_base_dataframe(snapshot, MigrationSimulator.REQUIRED_COLUMNS)

    # Calculate growth rates from historical data
    history = sim_data.sort_values(["Country", "Year"]).copy()
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

    # Example migration flows
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
    # Load configuration
    config = load_config(CONFIG_PATH)
    if config is None:
        raise RuntimeError("Failed to load configuration")
    
    # Get simulation parameters from config (with defaults)
    sim_config = config.get("simulation", {})
    init_prod = sim_config.get("initial_migrant_productivity", 0.5)
    prod_step = sim_config.get("productivity_step_per_year", 0.1)
    projection_years = sim_config.get("default_projection_years", 10)
    
    base_data, pop_growth, gdp_growth, migration_flows = build_example_inputs(config)

    simulator = MigrationSimulator(
        base_data,
        pop_proj=pop_growth,
        gdp_proj=gdp_growth,
        migration_flows=migration_flows,
        init_prod=init_prod,
        prod_step=prod_step,
    )
    end_year = simulator.year + projection_years
    results = simulator.run(end_year=end_year)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved example simulation covering up to {end_year} to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
