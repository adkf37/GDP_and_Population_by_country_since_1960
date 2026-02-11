import pandas as pd
import numpy as np
from migration_simulation import MigrationSimulator

# 1. Load and transform the base data from long to wide format
print("Loading and transforming data...")
raw_data = pd.read_csv("Data_GDP_Pop_by_Country_1960_Countries_only.csv")

# Clean the data - convert '..' to NaN and ensure numeric values
raw_data['Value'] = raw_data['Value'].replace('..', np.nan)
raw_data['Value'] = pd.to_numeric(raw_data['Value'], errors='coerce')

# Get the latest year data
latest_year = int(str(raw_data['Year'].max()).split()[0])  # Remove [YR2023] part
base_data = raw_data[raw_data['Year'] == raw_data['Year'].max()].copy()

# Pivot from long to wide format
base = base_data.pivot_table(
    index=['Country Name', 'Country Code'], 
    columns='Series Name', 
    values='Value'
).reset_index()

# Rename columns to match what MigrationSimulator expects
column_mapping = {
    'Country Name': 'Country',
    'GDP (current US$)': 'GDP',
    'Population, total': 'Total_Population'
}

# Only rename columns that exist
existing_mappings = {old: new for old, new in column_mapping.items() if old in base.columns}
base = base.rename(columns=existing_mappings)

# Add Year column
base['Year'] = latest_year

# Fill missing values with 0 for required columns and remove rows with missing critical data
if 'GDP' not in base.columns:
    print("Warning: GDP data not found, using zeros")
    base['GDP'] = 0.0
else:
    # Fill missing GDP values with 0
    base['GDP'] = base['GDP'].fillna(0.0)

if 'Total_Population' not in base.columns:
    print("Warning: Population data not found, using zeros") 
    base['Total_Population'] = 0.0
else:
    # Fill missing population values, but we'll filter out zero/negative values below
    base['Total_Population'] = base['Total_Population'].fillna(0.0)

# Remove rows with missing country names or zero/negative population
base = base.dropna(subset=['Country'])
base = base[base['Total_Population'] > 0]

# Also filter out rows where we don't have usable data
print(f"Filtering data: started with {len(base)} rows")
base = base.dropna(subset=['Country', 'GDP', 'Total_Population'])
print(f"After removing missing values: {len(base)} rows")

print(f"Prepared base data with {len(base)} countries for year {latest_year}")
print("Available columns:", base.columns.tolist())

# 2. Create example projection dictionaries and migration flows
# These are placeholder examples - replace with your actual projection data

# Example population growth projections (country, year) -> growth_rate
pop_proj = {}
for country in base['Country'].unique():
    for year in range(latest_year + 1, 2051):
        # Example: 1% annual population growth for all countries
        pop_proj[(country, year)] = 0.01

# Example GDP per capita growth projections
gdp_proj = {}
for country in base['Country'].unique():
    for year in range(latest_year + 1, 2051):
        # Example: 2% annual GDP per capita growth for all countries
        gdp_proj[(country, year)] = 0.02

# Example migration flows: year -> [(origin, destination, count), ...]
flows = {}
countries = base['Country'].tolist()
if len(countries) >= 2:
    for year in range(latest_year + 1, 2051):
        # Example: 10,000 people migrate from first country to second country each year
        flows[year] = [(countries[0], countries[1], 10000)]
else:
    flows = {}  # No migration if we don't have at least 2 countries

print(f"Created projections for {len(pop_proj)} country-year combinations")
print(f"Created migration flows for {len(flows)} years")

# 3. Run the simulation
print("Starting migration simulation...")
sim = MigrationSimulator(base, pop_proj, gdp_proj, flows,
                         init_prod=0.7, prod_step=0.1)
df_results = sim.run(end_year=2050)
df_results.to_csv("migration_sim_results.csv", index=False)
print("Simulation completed. Results saved to migration_sim_results.csv")
