import pandas as pd
from migration_simulation import MigrationSimulator

# 1. Load your base data
base = pd.read_csv("Data_GDP_Pop_by_Country_1960_Countries_only.csv")
base = base[base.Year == base.Year.max()]

# 2. Prepare projection dicts and migration flows
#    e.g. pop_proj = {('India',2025):0.01, ...}
#         gdp_proj = {('India',2025):0.025, ...}
#         flows    = {2025:[('India','US',200000), ...], ...}

sim = MigrationSimulator(base, pop_proj, gdp_proj, flows,
                         init_prod=0.7, prod_step=0.1)
df_results = sim.run(end_year=2050)
df_results.to_csv("migration_sim_results.csv", index=False)
