"""Vectorized migration simulation for improved performance.

This module provides a vectorized implementation of the MigrationSimulator
that uses pandas/numpy operations instead of Python loops for better
performance with large numbers of countries (200+).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from migration_simulation import (
    MigrationFlowsDict,
    ProjectionDict,
    validate_base_dataframe,
)

logger = logging.getLogger(__name__)


class VectorizedMigrationSimulator:
    """Vectorized migration simulator using pandas/numpy operations.
    
    This implementation is optimized for simulations with many countries
    by using vectorized operations instead of Python loops.
    """

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
        """Initialize vectorized simulator.
        
        Args:
            base_df: Base year data with Country, Year, GDP, Total_Population
            pop_proj: Population growth projections as (country, year) -> rate
            gdp_proj: GDP per capita growth projections as (country, year) -> rate
            migration_flows: Migration flows as year -> [(origin, dest, count), ...]
            init_prod: Initial migrant productivity (0-1)
            prod_step: Annual productivity increase
        """
        validate_base_dataframe(base_df, self.REQUIRED_COLUMNS)

        self.year = int(base_df["Year"].max())
        self.init_prod = float(init_prod)
        self.prod_step = float(prod_step)
        
        # Store countries as index for vectorized operations
        self.countries = base_df["Country"].tolist()
        self.country_idx = {c: i for i, c in enumerate(self.countries)}
        n = len(self.countries)
        
        # State as numpy arrays (faster than dict iteration)
        self.pop = np.zeros(n)
        self.gdp_pc = np.zeros(n)
        
        for _, row in base_df.iterrows():
            idx = self.country_idx[row["Country"]]
            pop = float(row["Total_Population"])
            if pop <= 0:
                raise ValueError(
                    f"Country '{row['Country']}' has non-positive population ({pop})"
                )
            self.pop[idx] = pop
            gdp_value = float(row["GDP"])
            self.gdp_pc[idx] = gdp_value / pop if gdp_value else 0.0
        
        # Convert projection dicts to arrays for vectorized lookup
        self.pop_proj = pop_proj
        self.gdp_proj = gdp_proj
        self.flows = {int(y): list(f) for y, f in migration_flows.items()}
        
        # Cohort tracking: DataFrame for vectorized operations
        # Each row is a cohort with: destination_idx, count, productivity
        self.cohort_data: List[Tuple[int, float, float]] = []
        
        self.history: List[Dict[str, float]] = []

    def _get_growth_rates(self, year: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get growth rate arrays for all countries for a given year.
        
        Returns:
            Tuple of (pop_growth, gdp_growth) arrays
        """
        pop_growth = np.zeros(len(self.countries))
        gdp_growth = np.zeros(len(self.countries))
        
        for i, country in enumerate(self.countries):
            pop_growth[i] = self.pop_proj.get((country, year), 0.0)
            gdp_growth[i] = self.gdp_proj.get((country, year), 0.0)
        
        return pop_growth, gdp_growth

    def step(self) -> None:
        """Advance simulation by one year using vectorized operations."""
        year = self.year + 1
        
        # Get growth rates (vectorized lookup could be optimized further with pre-computation)
        pop_growth, gdp_growth = self._get_growth_rates(year)
        
        # Apply growth rates (vectorized)
        self.pop = self.pop * (1 + pop_growth)
        self.gdp_pc = self.gdp_pc * (1 + gdp_growth)
        
        # Process migration flows
        for origin, destination, count in self.flows.get(year, []):
            if count <= 0:
                continue
            
            if origin not in self.country_idx:
                logger.warning("Origin country '%s' missing; skipping flow", origin)
                continue
            if destination not in self.country_idx:
                logger.warning("Destination country '%s' missing; skipping flow", destination)
                continue
            
            origin_idx = self.country_idx[origin]
            dest_idx = self.country_idx[destination]
            
            # Move migrants (clamp to available population)
            move = min(count, self.pop[origin_idx])
            if move <= 0:
                continue
            
            self.pop[origin_idx] -= move
            self.pop[dest_idx] += move
            
            # Track cohort for productivity adjustment
            self.cohort_data.append((dest_idx, move, self.init_prod))
        
        # Update cohort productivity (vectorized if many cohorts)
        updated_cohorts = []
        for dest_idx, count, prod in self.cohort_data:
            new_prod = min(1.0, prod + self.prod_step)
            updated_cohorts.append((dest_idx, count, new_prod))
        self.cohort_data = updated_cohorts
        
        # Calculate effective population and GDP (vectorized)
        shortfall = np.zeros(len(self.countries))
        for dest_idx, count, prod in self.cohort_data:
            shortfall[dest_idx] += (1 - prod) * count
        
        effective_pop = self.pop - shortfall
        gdp = effective_pop * self.gdp_pc
        
        # Record history
        year_record: Dict[str, float] = {"Year": float(year)}
        for i, country in enumerate(self.countries):
            year_record[f"{country}_pop"] = self.pop[i]
            year_record[f"{country}_gdp_pc"] = self.gdp_pc[i]
            year_record[f"{country}_gdp"] = gdp[i]
        
        self.history.append(year_record)
        self.year = year

    def run(self, end_year: int) -> pd.DataFrame:
        """Run simulation until end_year.
        
        Args:
            end_year: Year to stop simulation
            
        Returns:
            DataFrame with simulation history
        """
        while self.year < end_year:
            self.step()
        return pd.DataFrame(self.history)

    @property
    def state(self) -> Dict[str, Dict[str, float]]:
        """Get current state as dictionary (for compatibility)."""
        return {
            country: {"pop": self.pop[i], "gdp_pc": self.gdp_pc[i]}
            for i, country in enumerate(self.countries)
        }


class BatchProjectionLookup:
    """Pre-computed projection lookup for faster simulation.
    
    Converts sparse (country, year) -> value dictionaries to dense
    numpy arrays for faster vectorized lookups.
    """
    
    def __init__(
        self,
        projection: ProjectionDict,
        countries: List[str],
        start_year: int,
        end_year: int,
        default_value: float = 0.0,
    ) -> None:
        """Initialize batch lookup.
        
        Args:
            projection: Sparse projection dictionary
            countries: List of countries (defines row order)
            start_year: First year (inclusive)
            end_year: Last year (inclusive)
            default_value: Value for missing entries
        """
        self.countries = countries
        self.country_idx = {c: i for i, c in enumerate(countries)}
        self.start_year = start_year
        self.end_year = end_year
        
        n_countries = len(countries)
        n_years = end_year - start_year + 1
        
        # Create dense array: shape (n_countries, n_years)
        self.data = np.full((n_countries, n_years), default_value)
        
        for (country, year), value in projection.items():
            if country in self.country_idx and start_year <= year <= end_year:
                i = self.country_idx[country]
                j = year - start_year
                self.data[i, j] = value
    
    def get_year(self, year: int) -> np.ndarray:
        """Get all country values for a given year.
        
        Args:
            year: Year to lookup
            
        Returns:
            Array of values for all countries
        """
        if year < self.start_year or year > self.end_year:
            return np.zeros(len(self.countries))
        j = year - self.start_year
        return self.data[:, j]


class OptimizedMigrationSimulator(VectorizedMigrationSimulator):
    """Fully optimized simulator with pre-computed projections.
    
    Use this when running many simulations or very long time horizons.
    """
    
    def __init__(
        self,
        base_df: pd.DataFrame,
        pop_proj: ProjectionDict,
        gdp_proj: ProjectionDict,
        migration_flows: MigrationFlowsDict,
        init_prod: float = 0.5,
        prod_step: float = 0.1,
        end_year: Optional[int] = None,
    ) -> None:
        """Initialize optimized simulator with pre-computed lookups.
        
        Args:
            base_df: Base year data
            pop_proj: Population growth projections
            gdp_proj: GDP per capita growth projections
            migration_flows: Migration flows
            init_prod: Initial migrant productivity
            prod_step: Annual productivity increase
            end_year: Pre-compute lookups through this year (optional)
        """
        super().__init__(
            base_df, pop_proj, gdp_proj, migration_flows, init_prod, prod_step
        )
        
        # Determine year range for pre-computation
        start_year = self.year + 1
        if end_year is None:
            # Infer from projection keys
            all_years = set()
            for country, year in pop_proj.keys():
                all_years.add(year)
            for country, year in gdp_proj.keys():
                all_years.add(year)
            end_year = max(all_years) if all_years else start_year + 50
        
        # Pre-compute projection lookups
        self._pop_lookup = BatchProjectionLookup(
            pop_proj, self.countries, start_year, end_year
        )
        self._gdp_lookup = BatchProjectionLookup(
            gdp_proj, self.countries, start_year, end_year
        )

    def _get_growth_rates(self, year: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get growth rates using pre-computed batch lookup."""
        return self._pop_lookup.get_year(year), self._gdp_lookup.get_year(year)


__all__ = [
    "VectorizedMigrationSimulator",
    "OptimizedMigrationSimulator",
    "BatchProjectionLookup",
]
