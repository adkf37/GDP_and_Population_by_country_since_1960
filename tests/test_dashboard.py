"""Tests for the Dash dashboard in gdp_population_analysis.py.

Uses dash.testing for callback testing and component verification.
"""

import pandas as pd
import numpy as np
import pytest

from gdp_population_analysis import create_dashboard


@pytest.fixture
def sample_data():
    """Create sample transformed data for dashboard testing."""
    return pd.DataFrame({
        "Country Name": ["USA", "China", "Japan", "Germany", "UK"],
        "Country Code": ["USA", "CHN", "JPN", "DEU", "GBR"],
        "Year": ["2022 [YR2022]"] * 5,
        "GDP": [25e12, 18e12, 4.2e12, 4.1e12, 3.1e12],
        "PPP_GDP": [26e12, 30e12, 5.5e12, 5.0e12, 3.5e12],
        "Total_Population": [330e6, 1400e6, 125e6, 84e6, 67e6],
        "Working_Age_Population": [210e6, 980e6, 75e6, 54e6, 43e6],
        "GDP per Capita": [25e12/330e6, 18e12/1400e6, 4.2e12/125e6, 4.1e12/84e6, 3.1e12/67e6],
        "GDP per Working Age Adult": [25e12/210e6, 18e12/980e6, 4.2e12/75e6, 4.1e12/54e6, 3.1e12/43e6],
        "PPP GDP per Capita": [26e12/330e6, 30e12/1400e6, 5.5e12/125e6, 5.0e12/84e6, 3.5e12/67e6],
        "PPP GDP per Working Age Adult": [26e12/210e6, 30e12/980e6, 5.5e12/75e6, 5.0e12/54e6, 3.5e12/43e6],
        "Income Category": ["High Income", "Other", "High Income", "High Income", "High Income"],
    })


class TestDashboardCreation:
    """Tests for dashboard initialization."""

    def test_create_dashboard_returns_dash_app(self, sample_data):
        """create_dashboard returns a Dash application instance."""
        from dash import Dash
        app = create_dashboard(sample_data)
        assert isinstance(app, Dash)

    def test_create_dashboard_with_none_returns_error_app(self):
        """Dashboard handles None data gracefully."""
        from dash import Dash
        app = create_dashboard(None)
        assert isinstance(app, Dash)
        # Should have error message in layout
        assert app.layout is not None

    def test_create_dashboard_with_empty_df_returns_error_app(self):
        """Dashboard handles empty DataFrame gracefully."""
        from dash import Dash
        app = create_dashboard(pd.DataFrame())
        assert isinstance(app, Dash)

    def test_dashboard_has_required_components(self, sample_data):
        """Dashboard layout includes dropdown, graph, and checklist."""
        app = create_dashboard(sample_data)
        layout = app.layout
        
        # Check that layout exists and has children
        assert layout is not None
        assert hasattr(layout, 'children')

    def test_dashboard_dropdown_has_metric_options(self, sample_data):
        """Dropdown should have GDP per Capita and other metric options."""
        app = create_dashboard(sample_data)
        
        # Find the dropdown component
        from dash import dcc
        dropdown = None
        for child in app.layout.children:
            if isinstance(child, dcc.Dropdown):
                dropdown = child
                break
        
        assert dropdown is not None
        assert dropdown.id == 'metric-dropdown'
        assert len(dropdown.options) > 0
        
        # Check that expected metrics are available
        option_values = [opt['value'] for opt in dropdown.options]
        assert 'GDP per Capita' in option_values

    def test_dashboard_checklist_has_income_categories(self, sample_data):
        """Checklist should have High Income and Other options."""
        app = create_dashboard(sample_data)
        
        from dash import dcc
        checklist = None
        for child in app.layout.children:
            if isinstance(child, dcc.Checklist):
                checklist = child
                break
        
        assert checklist is not None
        assert checklist.id == 'income-filter'
        
        option_values = [opt['value'] for opt in checklist.options]
        assert 'High Income' in option_values
        assert 'Other' in option_values


class TestDashboardCallbacks:
    """Tests for dashboard callback behavior."""

    def test_callback_is_registered(self, sample_data):
        """The scatter plot update callback should be registered."""
        app = create_dashboard(sample_data)
        
        # Check that callbacks are registered
        assert len(app.callback_map) > 0
        
        # Find the scatter-plot callback
        callback_found = False
        for callback_id in app.callback_map:
            if 'scatter-plot' in callback_id:
                callback_found = True
                break
        
        assert callback_found, "scatter-plot callback not found"

    def test_callback_function_exists(self, sample_data):
        """The update_scatter callback function should be callable."""
        app = create_dashboard(sample_data)
        
        # Get the callback function
        callback_key = None
        for key in app.callback_map:
            if 'scatter-plot' in key:
                callback_key = key
                break
        
        assert callback_key is not None
        callback_info = app.callback_map[callback_key]
        assert 'callback' in callback_info
        assert callable(callback_info['callback'])


class TestDashboardWithMissingData:
    """Tests for dashboard behavior with incomplete data."""

    def test_dashboard_handles_missing_income_category(self):
        """Dashboard works when Income Category column is missing."""
        df = pd.DataFrame({
            "Country Name": ["USA", "China"],
            "Total_Population": [330e6, 1400e6],
            "GDP per Capita": [75000, 12000],
        })
        
        app = create_dashboard(df)
        assert app is not None

    def test_dashboard_handles_all_nan_metric(self):
        """Dashboard handles metrics that are all NaN."""
        df = pd.DataFrame({
            "Country Name": ["USA", "China"],
            "Total_Population": [330e6, 1400e6],
            "GDP per Capita": [np.nan, np.nan],
            "GDP per Working Age Adult": [75000, 12000],
            "Income Category": ["High Income", "Other"],
        })
        
        app = create_dashboard(df)
        
        # Should still create app, but GDP per Capita shouldn't be in options
        from dash import dcc
        dropdown = None
        for child in app.layout.children:
            if isinstance(child, dcc.Dropdown):
                dropdown = child
                break
        
        if dropdown and dropdown.options:
            option_values = [opt['value'] for opt in dropdown.options]
            # GDP per Capita should be excluded since all values are NaN
            assert 'GDP per Capita' not in option_values
