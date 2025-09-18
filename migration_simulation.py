"""Simulation utilities for exploring migration-driven macro scenarios.

This module exposes the :class:`MigrationSimulator` – a light-weight engine that
tracks country level population and GDP per capita trajectories – along with a
set of helpers that make it easier to transform raw projection files into the
dictionary inputs required by the simulator.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ProjectionDict = Dict[Tuple[str, int], float]
MigrationFlowsDict = Dict[int, List[Tuple[str, str, float]]]


class MigrationSimulator:
    """Simulates country-level population and GDP trajectories under migration scenarios."""

    REQUIRED_COLUMNS = {"Country", "Year", "Total_Population", "GDP"}

    def __init__(
        self,
        base_df: pd.DataFrame,
        pop_proj: ProjectionDict,
        gdp_proj: ProjectionDict,
        migration_flows: MigrationFlowsDict,
        init_prod: float = 0.5,
        prod_step: float = 0.1,
    ) -> None:
        validate_base_dataframe(base_df, self.REQUIRED_COLUMNS)

        self.year = int(base_df["Year"].max())
        self.state: Dict[str, Dict[str, float]] = {}
        for _, row in base_df.iterrows():
            country = row["Country"]
            pop = float(row["Total_Population"])
            if pop <= 0:
                raise ValueError(
                    f"Country '{country}' has non-positive population ({pop}). "
                    "Ensure base data represents living population counts."
                )
            gdp_value = float(row["GDP"])
            gdp_per_cap = gdp_value / pop if gdp_value else 0.0
            self.state[country] = {"pop": pop, "gdp_pc": gdp_per_cap}

        self.pop_proj = dict(pop_proj)
        self.gdp_proj = dict(gdp_proj)
        self.flows = {int(year): list(flows) for year, flows in migration_flows.items()}
        self.cohorts: Dict[str, List[Dict[str, float]]] = {c: [] for c in self.state}
        self.init_prod = float(init_prod)
        self.prod_step = float(prod_step)
        self.history: List[Dict[str, float]] = []

    def step(self) -> None:
        year = self.year + 1
        new_state: Dict[str, Dict[str, float]] = {}

        for country, state in self.state.items():
            pop_growth = self.pop_proj.get((country, year), 0.0)
            gdp_growth = self.gdp_proj.get((country, year), 0.0)
            pop = state["pop"] * (1 + pop_growth)
            gdp_pc = state["gdp_pc"] * (1 + gdp_growth)
            new_state[country] = {"pop": pop, "gdp_pc": gdp_pc}

        for origin, destination, count in self.flows.get(year, []):
            if count <= 0:
                logger.debug(
                    "Skipping non-positive migration flow %s -> %s: %s",
                    origin,
                    destination,
                    count,
                )
                continue

            if origin not in new_state:
                logger.warning("Origin country '%s' missing from state; skipping flow", origin)
                continue
            if destination not in new_state:
                logger.warning(
                    "Destination country '%s' missing from state; skipping flow",
                    destination,
                )
                continue

            available = new_state[origin]["pop"]
            move = min(count, available)
            if move <= 0:
                continue

            new_state[origin]["pop"] -= move
            new_state[destination]["pop"] += move
            self.cohorts.setdefault(destination, [])
            self.cohorts[destination].append({"year": year, "count": move, "prod": self.init_prod})

        year_record: Dict[str, float] = {"Year": year}
        for country, state in new_state.items():
            for cohort in self.cohorts[country]:
                cohort["prod"] = min(1.0, cohort["prod"] + self.prod_step)

            shortfall = sum((1 - cohort["prod"]) * cohort["count"] for cohort in self.cohorts[country])
            effective_population = state["pop"] - shortfall
            gdp = effective_population * state["gdp_pc"]

            year_record[f"{country}_pop"] = state["pop"]
            year_record[f"{country}_gdp_pc"] = state["gdp_pc"]
            year_record[f"{country}_gdp"] = gdp

        self.history.append(year_record)
        self.state = new_state
        self.year = year

    def run(self, end_year: int) -> pd.DataFrame:
        while self.year < end_year:
            self.step()
        return pd.DataFrame(self.history)


def validate_base_dataframe(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            "Base dataframe is missing required columns: " + ", ".join(sorted(missing))
        )

    if df["Country"].isna().any():
        raise ValueError("Base dataframe contains rows with missing country names.")

    if df["GDP"].isna().any():
        raise ValueError("Base dataframe contains rows with missing GDP values.")

    if df["Total_Population"].isna().any():
        raise ValueError("Base dataframe contains rows with missing population values.")


def dataframe_to_projection_dict(
    df: pd.DataFrame,
    *,
    country_col: str,
    year_col: str,
    value_col: str,
    value_transform: Optional[Callable[[float], float]] = None,
    dropna: bool = True,
) -> ProjectionDict:
    if value_transform is None:
        value_transform = lambda value: value  # type: ignore[return-value]

    projection: ProjectionDict = {}
    for _, row in df.iterrows():
        country = row[country_col]
        if not isinstance(country, str) or not country.strip():
            continue

        year_raw = row[year_col]
        if pd.isna(year_raw):
            continue
        year = int(year_raw)

        value = row[value_col]
        if dropna and pd.isna(value):
            continue

        numeric_value = float(value_transform(float(value)))
        projection[(country, year)] = numeric_value
    return projection


def load_projection_csv(
    path: str,
    *,
    country_col: str,
    year_col: str,
    value_col: str,
    filters: Optional[Mapping[str, Iterable]] = None,
    value_transform: Optional[Callable[[float], float]] = None,
    dropna: bool = True,
    read_csv_kwargs: Optional[Mapping[str, object]] = None,
) -> ProjectionDict:
    kwargs = dict(read_csv_kwargs or {})
    df = pd.read_csv(path, **kwargs)
    if filters:
        for column, allowed in filters.items():
            df = df[df[column].isin(allowed)]
    return dataframe_to_projection_dict(
        df,
        country_col=country_col,
        year_col=year_col,
        value_col=value_col,
        value_transform=value_transform,
        dropna=dropna,
    )


def migration_flows_from_dataframe(
    df: pd.DataFrame,
    *,
    origin_col: str,
    destination_col: str,
    year_col: str,
    flow_col: str,
    dropna: bool = True,
) -> MigrationFlowsDict:
    flows: MigrationFlowsDict = {}
    for _, row in df.iterrows():
        origin = row[origin_col]
        destination = row[destination_col]
        count = row[flow_col]
        year_raw = row[year_col]

        if dropna and (
            pd.isna(origin)
            or pd.isna(destination)
            or pd.isna(count)
            or pd.isna(year_raw)
        ):
            continue

        if not isinstance(origin, str) or not isinstance(destination, str):
            continue

        year = int(year_raw)
        count_value = float(count)
        if count_value < 0:
            raise ValueError("Migration counts must be non-negative.")

        flows.setdefault(year, []).append((origin, destination, count_value))
    return flows


def load_migration_flows_csv(
    path: str,
    *,
    origin_col: str,
    destination_col: str,
    year_col: str,
    flow_col: str,
    dropna: bool = True,
    read_csv_kwargs: Optional[Mapping[str, object]] = None,
) -> MigrationFlowsDict:
    kwargs = dict(read_csv_kwargs or {})
    df = pd.read_csv(path, **kwargs)
    return migration_flows_from_dataframe(
        df,
        origin_col=origin_col,
        destination_col=destination_col,
        year_col=year_col,
        flow_col=flow_col,
        dropna=dropna,
    )


__all__ = [
    "MigrationSimulator",
    "ProjectionDict",
    "MigrationFlowsDict",
    "dataframe_to_projection_dict",
    "load_projection_csv",
    "migration_flows_from_dataframe",
    "load_migration_flows_csv",
    "validate_base_dataframe",
]
