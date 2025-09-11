import pandas as pd
import numpy as np

class MigrationSimulator:
    """
    Simulates country-level population and GDP trajectories under migration scenarios.
    """
    def __init__(self,
                 base_df: pd.DataFrame,
                 pop_proj: dict,      # {(country, year): pop_growth_rate}
                 gdp_proj: dict,      # {(country, year): gdp_per_cap_growth_rate}
                 migration_flows: dict,# {year: [(origin, dest, count), ...]}
                 init_prod: float=0.5,
                 prod_step: float=0.1):
        # Base year is the max Year in base_df
        self.year = int(base_df['Year'].max())
        self.state = {row['Country']: {
                          'pop': row['Total_Population'],
                          'gdp_pc': row['GDP'] / row['Total_Population']
                      } for _, row in base_df.iterrows()}
        self.pop_proj = pop_proj
        self.gdp_proj = gdp_proj
        self.flows   = migration_flows
        self.cohorts = {c: [] for c in self.state}
        self.init_prod   = init_prod
        self.prod_step   = prod_step
        self.history = []

    def step(self):
        y = self.year + 1
        new_state = {}
        # 1) Natural growth
        for c, st in self.state.items():
            pg = self.pop_proj.get((c, y), 0.0)
            gpg = self.gdp_proj.get((c, y), 0.0)
            pop = st['pop'] * (1 + pg)
            gdp_pc = st['gdp_pc'] * (1 + gpg)
            new_state[c] = {'pop': pop, 'gdp_pc': gdp_pc}
        # 2) Migration flows
        for orig, dest, cnt in self.flows.get(y, []):
            cnt = min(cnt, new_state[orig]['pop'])
            new_state[orig]['pop'] -= cnt
            new_state[dest]['pop'] += cnt
            self.cohorts[dest].append({'year': y, 'count': cnt, 'prod': self.init_prod})
        # 3) Update cohorts’ productivity and compute GDP
        year_rec = {'Year': y}
        for c, st in new_state.items():
            # update cohort productivity
            for coh in self.cohorts[c]:
                coh['prod'] = min(1.0, coh['prod'] + self.prod_step)
            # effective pop = pop − ∑(1−prod)*count
            shortfall = sum((1-coh['prod'])*coh['count'] for coh in self.cohorts[c])
            eff_pop = st['pop'] - shortfall
            gdp = eff_pop * st['gdp_pc']
            year_rec[f'{c}_pop'] = st['pop']
            year_rec[f'{c}_gdp_pc'] = st['gdp_pc']
            year_rec[f'{c}_gdp'] = gdp
        self.history.append(year_rec)
        self.state = new_state
        self.year = y

    def run(self, end_year):
        while self.year < end_year:
            self.step()
        return pd.DataFrame(self.history)
