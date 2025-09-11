# run_migration_simulation.py
import pandas as pd
from migration_simulation import MigrationSimulator

# Load base data (historical latest year) from the repository CSV
base_df = pd.read_csv("Data_GDP_Pop_by_Country_1960_Countries_only.csv")
base_year = 2023  # for example, if 2020 is last year in data
base_data = base_df[base_df['Year'] == base_year]  # filter latest year

# Load or define projection inputs
pop_growth = pd.read_csv(
    "https://population.un.org/wpp/Download/Files/1_Indicators%20of%20Population/WPP2022_POPPROJ1DT.csv"
)
gdp_growth = pd.read_csv( 
    "https://www.imf.org/external/datamapper/export/csv.php?indicator=NGDPDPC"  # function or manual input

)
migration_flows = load_migration_flows("migration_flows.csv")    # or define a dictionary
integration_params = {"init_productivity": 0.5, "annual_increase": 0.1}

# Initialize simulator
sim = MigrationSimulator(base_data, pop_growth, gdp_growth, migration_flows, integration_params)
# Run simulation from base_year+1 to 2050
results_df = sim.run(end_year=2050)

# Save or print results
results_df.to_csv("simulation_results_2020_2050.csv", index=False)
print("Global GDP in 2050:", results_df[results_df['Year']==2050]['GDP'].sum())
