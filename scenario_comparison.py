"""Scenario comparison utility for migration simulations.

This module provides tools for running multiple migration scenarios
and comparing their outcomes side-by-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from migration_simulation import (
    MigrationSimulator,
    MigrationFlowsDict,
    ProjectionDict,
)


@dataclass
class ScenarioConfig:
    """Configuration for a single migration scenario."""
    
    name: str
    migration_flows: MigrationFlowsDict
    init_prod: float = 0.5
    prod_step: float = 0.1
    description: str = ""


@dataclass
class ScenarioResult:
    """Results from running a single scenario."""
    
    name: str
    description: str
    history: pd.DataFrame
    final_year: int
    final_state: Dict[str, Dict[str, float]]
    
    def get_country_trajectory(self, country: str, metric: str = "gdp") -> pd.Series:
        """Get time series for a specific country and metric.
        
        Args:
            country: Country name
            metric: One of 'gdp', 'pop', 'gdp_pc'
            
        Returns:
            Series indexed by year
        """
        col_name = f"{country}_{metric}"
        if col_name not in self.history.columns:
            available = [c for c in self.history.columns if c.startswith(f"{country}_")]
            raise KeyError(f"Column '{col_name}' not found. Available: {available}")
        return self.history.set_index("Year")[col_name]

    def get_final_gdp(self, country: str) -> float:
        """Get final year GDP for a country."""
        return self.history[f"{country}_gdp"].iloc[-1]

    def get_final_population(self, country: str) -> float:
        """Get final year population for a country."""
        return self.history[f"{country}_pop"].iloc[-1]


class ScenarioRunner:
    """Runs and compares multiple migration scenarios."""
    
    def __init__(
        self,
        base_df: pd.DataFrame,
        pop_proj: ProjectionDict,
        gdp_proj: ProjectionDict,
        end_year: int,
    ) -> None:
        """Initialize scenario runner.
        
        Args:
            base_df: Base year data with Country, Year, GDP, Total_Population
            pop_proj: Population growth projections
            gdp_proj: GDP per capita growth projections
            end_year: Year to project until
        """
        self.base_df = base_df.copy()
        self.pop_proj = dict(pop_proj)
        self.gdp_proj = dict(gdp_proj)
        self.end_year = end_year
        self.results: Dict[str, ScenarioResult] = {}
    
    def run_scenario(self, config: ScenarioConfig) -> ScenarioResult:
        """Run a single scenario and store results.
        
        Args:
            config: Scenario configuration
            
        Returns:
            ScenarioResult with simulation history
        """
        simulator = MigrationSimulator(
            self.base_df.copy(),
            pop_proj=self.pop_proj,
            gdp_proj=self.gdp_proj,
            migration_flows=config.migration_flows,
            init_prod=config.init_prod,
            prod_step=config.prod_step,
        )
        
        history = simulator.run(end_year=self.end_year)
        
        result = ScenarioResult(
            name=config.name,
            description=config.description,
            history=history,
            final_year=simulator.year,
            final_state=dict(simulator.state),
        )
        
        self.results[config.name] = result
        return result
    
    def run_scenarios(self, configs: List[ScenarioConfig]) -> Dict[str, ScenarioResult]:
        """Run multiple scenarios.
        
        Args:
            configs: List of scenario configurations
            
        Returns:
            Dictionary mapping scenario names to results
        """
        for config in configs:
            self.run_scenario(config)
        return self.results
    
    def compare_metric(
        self,
        country: str,
        metric: str = "gdp",
        scenarios: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare a metric across scenarios for a specific country.
        
        Args:
            country: Country name
            metric: One of 'gdp', 'pop', 'gdp_pc'
            scenarios: List of scenario names (None = all)
            
        Returns:
            DataFrame with years as index, scenarios as columns
        """
        if scenarios is None:
            scenarios = list(self.results.keys())
        
        comparison = pd.DataFrame()
        for name in scenarios:
            if name not in self.results:
                continue
            result = self.results[name]
            try:
                trajectory = result.get_country_trajectory(country, metric)
                comparison[name] = trajectory
            except KeyError:
                continue
        
        return comparison
    
    def compare_final_values(
        self,
        countries: List[str],
        metric: str = "gdp",
        scenarios: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare final year values across countries and scenarios.
        
        Args:
            countries: List of countries to compare
            metric: One of 'gdp', 'pop', 'gdp_pc'
            scenarios: List of scenario names (None = all)
            
        Returns:
            DataFrame with countries as index, scenarios as columns
        """
        if scenarios is None:
            scenarios = list(self.results.keys())
        
        data = {}
        for name in scenarios:
            if name not in self.results:
                continue
            result = self.results[name]
            values = {}
            for country in countries:
                col = f"{country}_{metric}"
                if col in result.history.columns:
                    values[country] = result.history[col].iloc[-1]
                else:
                    values[country] = np.nan
            data[name] = values
        
        return pd.DataFrame(data)
    
    def calculate_differences(
        self,
        baseline_scenario: str,
        countries: List[str],
        metric: str = "gdp",
    ) -> pd.DataFrame:
        """Calculate differences from baseline scenario.
        
        Args:
            baseline_scenario: Name of the baseline scenario
            countries: List of countries to compare
            metric: One of 'gdp', 'pop', 'gdp_pc'
            
        Returns:
            DataFrame with differences from baseline (absolute and %)
        """
        if baseline_scenario not in self.results:
            raise ValueError(f"Baseline scenario '{baseline_scenario}' not found")
        
        final_values = self.compare_final_values(countries, metric)
        baseline = final_values[baseline_scenario]
        
        diffs = pd.DataFrame()
        for scenario in final_values.columns:
            if scenario == baseline_scenario:
                continue
            diffs[f"{scenario}_diff"] = final_values[scenario] - baseline
            diffs[f"{scenario}_pct"] = (final_values[scenario] - baseline) / baseline * 100
        
        return diffs

    def summary_report(self, countries: Optional[List[str]] = None) -> str:
        """Generate a text summary of all scenario results.
        
        Args:
            countries: Countries to include (None = use first 5 from results)
            
        Returns:
            Formatted summary string
        """
        if not self.results:
            return "No scenarios have been run yet."
        
        # Get countries from first result if not specified
        if countries is None:
            first_result = list(self.results.values())[0]
            gdp_cols = [c for c in first_result.history.columns if c.endswith("_gdp")]
            countries = [c.replace("_gdp", "") for c in gdp_cols[:5]]
        
        lines = ["=" * 60, "SCENARIO COMPARISON SUMMARY", "=" * 60, ""]
        
        for name, result in self.results.items():
            lines.append(f"Scenario: {name}")
            if result.description:
                lines.append(f"  Description: {result.description}")
            lines.append(f"  Final Year: {result.final_year}")
            lines.append("")
            
            for country in countries:
                gdp_col = f"{country}_gdp"
                pop_col = f"{country}_pop"
                if gdp_col in result.history.columns:
                    final_gdp = result.history[gdp_col].iloc[-1]
                    final_pop = result.history[pop_col].iloc[-1] if pop_col in result.history.columns else None
                    lines.append(f"  {country}:")
                    lines.append(f"    GDP: ${final_gdp:,.0f}")
                    if final_pop:
                        lines.append(f"    Population: {final_pop:,.0f}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")
        
        return "\n".join(lines)


def create_baseline_scenario() -> ScenarioConfig:
    """Create a baseline scenario with no migration."""
    return ScenarioConfig(
        name="baseline",
        migration_flows={},
        description="No migration flows - baseline projection",
    )


def create_migration_shock_scenario(
    name: str,
    origin: str,
    destination: str,
    annual_migrants: float,
    start_year: int,
    end_year: int,
    description: str = "",
) -> ScenarioConfig:
    """Create a scenario with constant annual migration between two countries.
    
    Args:
        name: Scenario name
        origin: Origin country
        destination: Destination country
        annual_migrants: Number of migrants per year
        start_year: First year of migration
        end_year: Last year of migration
        description: Scenario description
        
    Returns:
        ScenarioConfig
    """
    flows: MigrationFlowsDict = {}
    for year in range(start_year, end_year + 1):
        flows[year] = [(origin, destination, annual_migrants)]
    
    if not description:
        description = f"{annual_migrants:,.0f} migrants/year from {origin} to {destination}"
    
    return ScenarioConfig(
        name=name,
        migration_flows=flows,
        description=description,
    )


__all__ = [
    "ScenarioConfig",
    "ScenarioResult",
    "ScenarioRunner",
    "create_baseline_scenario",
    "create_migration_shock_scenario",
]
