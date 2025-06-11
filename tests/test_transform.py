import pytest
import pandas as pd
from gdp_population_analysis import clean_and_transform_data


def test_gdp_and_ppp_per_capita():
    df = pd.DataFrame({
        'GDP': [1000, 2000],
        'PPP_GDP': [1100, 2100],
        'Working_Age_Population': [100, 200],
        'Total_Population': [150, 250]
    })

    result = clean_and_transform_data(df.copy(), 80)

    expected_gdp_pc = [1000 / 150, 2000 / 250]
    expected_ppp_pc = [1100 / 150, 2100 / 250]

    assert result['GDP per Capita'].tolist() == pytest.approx(expected_gdp_pc)
    assert result['PPP GDP per Capita'].tolist() == pytest.approx(expected_ppp_pc)
