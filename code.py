import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json  # Added for loading config

CONFIG_FILE_PATH = 'config.json'


def load_config(config_path):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_path}'.")
        return None


def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        # data.head() # Still commented out
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Data file '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        return None


def clean_and_transform_data(df, high_income_threshold):
    """Cleans, transforms, and enriches the dataframe."""
    if df is None:
        return None

    # Replace missing or placeholder values with NaN
    df.replace(['..', 'N/A', ''], np.nan, inplace=True)

    # Convert relevant columns to numeric where possible
    for col in ['GDP', 'PPP_GDP', 'Working_Age_Population', 'Total_Population']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate additional metrics
    df['GDP per Capita'] = df['GDP'] / df['Total_Population']
    df['GDP per Working Age Adult'] = df['GDP'] / df['Working_Age_Population']
    df['PPP GDP per Capita'] = df['PPP_GDP'] / df['Total_Population']
    df['PPP GDP per Working Age Adult'] = df['PPP_GDP'] / df['Working_Age_Population']

    # Define high-income countries
    threshold_value = np.percentile(df['GDP per Capita'].dropna(), high_income_threshold)
    df['Income Category'] = np.where(df['GDP per Capita'] >= threshold_value, 'High Income', 'Other')
    return df


def perform_exploratory_analysis(df):
    """Performs and prints exploratory data analysis results."""
    if df is None:
        return

    # Correlation
    correlation = df[['GDP', 'GDP per Working Age Adult']].corr().iloc[0, 1]
    print(f"Correlation between GDP and GDP per Working Age Adult: {correlation}")

    # Static Plot (Matplotlib)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['GDP per Capita'], df['GDP per Working Age Adult'], alpha=0.7,
                c=df['Income Category'].map({'High Income': 'blue', 'Other': 'orange'}))
    plt.title('GDP per Capita vs GDP per Working Age Adult')
    plt.xlabel('GDP per Capita')
    plt.ylabel('GDP per Working Age Adult')
    plt.grid(True)
    # plt.colorbar(label='Income Category') # colorbar might error if one category is missing after filtering
    # plt.show() # Still commented out

    # Aggregated Insights
    grouped_data = df.groupby('Income Category').agg(
        Average_GDP=('GDP', 'mean'),
        Average_GDP_per_Working_Age=('GDP per Working Age Adult', 'mean'),
        Average_GDP_per_Capita=('GDP per Capita', 'mean'),
        Population=('Total_Population', 'sum')
    ).reset_index()
    print("\nAggregated Data by Income Category:")
    print(grouped_data)


def create_dashboard(df_initial):
    """Creates and configures the Dash application."""
    if df_initial is None:
        print("Cannot create dashboard without data.")
        return None

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('GDP and Population Analysis Dashboard', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'GDP per Capita', 'value': 'GDP per Capita'},
                {'label': 'GDP per Working Age Adult', 'value': 'GDP per Working Age Adult'},
                {'label': 'PPP GDP per Capita', 'value': 'PPP GDP per Capita'},
                {'label': 'PPP GDP per Working Age Adult', 'value': 'PPP GDP per Working Age Adult'}
            ],
            value='GDP per Capita',
            placeholder='Select a metric to visualize'
        ),
        dcc.Graph(id='scatter-plot'),
        html.P('Select Income Category:'),
        dcc.Checklist(
            id='income-filter',
            options=[
                {'label': 'High Income', 'value': 'High Income'},
                {'label': 'Other', 'value': 'Other'}
            ],
            value=['High Income', 'Other'],  # Default to all categories
            inline=True
        )
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('metric-dropdown', 'value'),
        Input('income-filter', 'value')
    )
    def update_scatter(metric, income_filter_values):
        # df_initial is the full dataset loaded at the start
        # We filter it here based on the callback inputs
        if not income_filter_values:  # Handle case where no checkboxes are selected
            return px.scatter(title=f"No income categories selected. Please select at least one.")

        filtered_data = df_initial[df_initial['Income Category'].isin(income_filter_values)]

        if filtered_data.empty:
            return px.scatter(
                title=f"No data for selected metric '{metric}' and income categories: {', '.join(income_filter_values)}")

        fig = px.scatter(
            filtered_data,
            x='Total_Population',
            y=metric,
            color='Income Category',
            hover_name='Country Name',
            title=f'{metric} vs Total Population'
        )
        return fig

    return app


# Main execution
if __name__ == '__main__':
    config = load_config(CONFIG_FILE_PATH)

    if config:
        data_file = config.get('data_file_path')
        high_income_percentile = config.get('high_income_percentile_threshold', 80)  # Default to 80 if not in config

        if data_file:
            raw_data = load_data(data_file)

            if raw_data is not None:
                transformed_data = clean_and_transform_data(raw_data.copy(), high_income_percentile)  # Use .copy() to avoid modifying original raw_data

                if transformed_data is not None:
                    perform_exploratory_analysis(transformed_data)
                    dashboard_app = create_dashboard(transformed_data)  # Pass transformed_data to the dashboard

                    if dashboard_app:
                        dashboard_app.run_server(debug=True)
                else:
                    print("Data cleaning and transformation failed.")
            else:
                print("Data loading failed.")
        else:
            print("Data file path not found in configuration.")
    else:
        print("Configuration loading failed. Exiting.")
