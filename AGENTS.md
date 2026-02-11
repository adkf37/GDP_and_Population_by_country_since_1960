# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Build and Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main analysis dashboard (launches Dash app at http://localhost:8050)
python gdp_population_analysis.py

# Run migration simulation scenarios
python run_migration_simulation.py

# Run all tests
pytest -q

# Run a specific test file
pytest tests/test_migration_helpers.py -v

# Run a specific test function
pytest tests/test_migration_helpers.py::test_dataframe_to_projection_dict_basic -v
```

## Architecture Overview

### Core Components

**Configuration (`config.json`)**
Controls data loading behavior: file paths, column mappings for pivoting long-format CSV data to wide format, percentile threshold for categorizing high-income countries, and simulation parameters. See `CONFIG_SCHEMA.md` for full documentation.

**Shared Data Loader (`data_loader.py`)**
- `load_and_transform()`: One-call function to load CSV, reshape to wide format, calculate derived metrics
- `prepare_for_simulation()`: Prepares DataFrame for `MigrationSimulator` (renames columns, parses years, filters)
- `check_data_completeness()`: Reports missing values by country, year, and series
- `parse_year()`: Handles World Bank year format (`"2022 [YR2022]"`) conversion

**Exploratory Analysis (`gdp_population_analysis.py`)**
- Loads and reshapes raw CSV from long format (one row per series/year) to wide format (one row per country/year with columns for GDP, Population, etc.)
- Calculates derived metrics: GDP per Capita, GDP per Working Age Adult, PPP variants
- Categorizes countries by income level using percentile threshold
- Exposes an interactive Dash dashboard with metric selection and income filtering

**Migration Simulation Engine (`migration_simulation.py`)**
- `MigrationSimulator`: Core class that tracks country-level population and GDP trajectories. Requires a base DataFrame with columns `Country`, `Year`, `Total_Population`, `GDP`.
- `ProjectionDict` type: `Dict[Tuple[str, int], float]` mapping (country, year) to growth rates
- `MigrationFlowsDict` type: `Dict[int, List[Tuple[str, str, float]]]` mapping year to list of (origin, destination, count) tuples
- Helper functions (`dataframe_to_projection_dict`, `migration_flows_from_dataframe`) convert CSVs to these dict formats

**Vectorized Simulation (`migration_simulation_vectorized.py`)**
- `VectorizedMigrationSimulator`: Numpy-based implementation for better performance with 200+ countries
- `OptimizedMigrationSimulator`: Pre-computes projection lookups for long time horizons
- `BatchProjectionLookup`: Converts sparse projection dicts to dense numpy arrays

**Scenario Comparison (`scenario_comparison.py`)**
- `ScenarioRunner`: Run and compare multiple migration scenarios
- `ScenarioConfig`/`ScenarioResult`: Configuration and result dataclasses
- `create_baseline_scenario()`, `create_migration_shock_scenario()`: Helper factories

**Simulation Entry Points**
- `run_migration_simulation.py`: Example driver that builds inputs from historical data and runs a 10-year projection
- `run_sim.py`: Standalone script demonstrating end-to-end data loading, transformation, and simulation through 2050

### Data Flow

1. Raw CSV (long format with `Country Name`, `Year`, `Series Name`, `Value` columns) →
2. `reshape_data()` pivots to wide format, renames columns per `config.json` mapping →
3. `clean_and_transform_data()` converts to numeric, calculates derived metrics →
4. DataFrame feeds into dashboard or `MigrationSimulator`

### Test Structure

Tests in `tests/` use pytest. `conftest.py` adds the repo root to `sys.path` so imports work without package installation.
- `test_migration_helpers.py`: Tests for projection dict conversion, migration flow parsing, and simulator validation/edge cases
- `test_transform.py`: Tests for data transformation and derived metric calculations
