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
        print(f"Successfully loaded {file_path}. Columns: {data.columns.tolist()}")
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

def reshape_data(df, pivot_config, series_mapping):
    """Reshapes data from long to wide format and renames columns."""
    if df is None:
        return None
    try:
        # Clean the 'Value' column before pivoting
        value_col = pivot_config['pivot_values_column']
        if value_col in df.columns:
            print(f"Cleaning and converting column '{value_col}' to numeric before pivoting...")
            df.replace({'..': np.nan, 'N/A': np.nan, '': np.nan}, inplace=True) # More targeted replace
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            print(f"Finished cleaning '{value_col}'.")
        else:
            print(f"Error: Pivot values column '{value_col}' not found in DataFrame.")
            return None

        print("Pivoting data...")
        df_pivoted = df.pivot_table(
            index=pivot_config['pivot_index_columns'],
            columns=pivot_config['pivot_columns_column'],
            values=pivot_config['pivot_values_column']
        ).reset_index()
        print(f"Pivoted data columns: {df_pivoted.columns.tolist()}")
        
        # Rename columns based on the mapping
        df_pivoted.rename(columns=series_mapping, inplace=True)
        print(f"Renamed columns: {df_pivoted.columns.tolist()}")
        
        # Ensure all expected columns from the mapping are present, fill with NaN if not
        for expected_col in series_mapping.values():
            if expected_col not in df_pivoted.columns:
                df_pivoted[expected_col] = np.nan
                print(f"Added missing expected column: {expected_col}")

        return df_pivoted
    except KeyError as e:
        print(f"Error during pivoting/reshaping: Missing key {e}. Check pivot_config and CSV columns.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data reshaping: {e}")
        return None

def clean_and_transform_data(df, high_income_threshold):
    """Cleans, transforms, and enriches the dataframe."""
    if df is None:
        return None
    
    # Replace missing or placeholder values with NaN - this might have been done by pivot for non-existent series
    # df.replace(['..', 'N/A', ''], np.nan, inplace=True) # Values should be numeric or NaN after pivot

    # Columns to convert to numeric are the values from the series_mapping
    numeric_cols = ['GDP', 'PPP_GDP', 'Working_Age_Population', 'Total_Population'] # These are the target names

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected column '{col}' not found for numeric conversion. It will be missing in calculations.")
            df[col] = np.nan # Ensure column exists for subsequent calculations to avoid KeyErrors

    # Calculate additional metrics
    # Add checks for column existence before calculation
    if 'GDP' in df.columns and 'Total_Population' in df.columns:
        df['GDP per Capita'] = df['GDP'] / df['Total_Population']
    else:
        df['GDP per Capita'] = np.nan

    if 'GDP' in df.columns and 'Working_Age_Population' in df.columns:
        df['GDP per Working Age Adult'] = df['GDP'] / df['Working_Age_Population']
    else:
        df['GDP per Working Age Adult'] = np.nan
        
    if 'PPP_GDP' in df.columns and 'Total_Population' in df.columns:
        df['PPP GDP per Capita'] = df['PPP_GDP'] / df['Total_Population']
    else:
        df['PPP GDP per Capita'] = np.nan

    if 'PPP_GDP' in df.columns and 'Working_Age_Population' in df.columns:
        df['PPP GDP per Working Age Adult'] = df['PPP_GDP'] / df['Working_Age_Population']
    else:
        df['PPP GDP per Working Age Adult'] = np.nan

    # Define high-income countries
    if 'GDP per Capita' in df.columns and not df['GDP per Capita'].isnull().all():
        threshold_value = np.percentile(df['GDP per Capita'].dropna(), high_income_threshold)
        df['Income Category'] = np.where(df['GDP per Capita'] >= threshold_value, 'High Income', 'Other')
    else:
        print("Warning: 'GDP per Capita' is missing or all NaN. Cannot define 'Income Category'. Setting to 'Other'.")
        df['Income Category'] = 'Other'
        
    return df

def perform_exploratory_analysis(df):
    """Performs and prints exploratory data analysis results."""
    if df is None:
        return

    # Correlation
    if 'GDP' in df.columns and 'GDP per Working Age Adult' in df.columns and \
       not df['GDP'].isnull().all() and not df['GDP per Working Age Adult'].isnull().all():
        correlation = df[['GDP', 'GDP per Working Age Adult']].corr().iloc[0, 1]
        print(f"Correlation between GDP and GDP per Working Age Adult: {correlation}")
    else:
        print("Cannot calculate correlation due to missing or all-NaN GDP or GDP per Working Age Adult columns.")

    # Static Plot (Matplotlib)
    if 'GDP per Capita' in df.columns and 'GDP per Working Age Adult' in df.columns and 'Income Category' in df.columns:
        plt.figure(figsize=(10, 6))
        # Filter out NaNs for plotting
        plot_df = df[['GDP per Capita', 'GDP per Working Age Adult', 'Income Category']].dropna()
        if not plot_df.empty:
            plt.scatter(plot_df['GDP per Capita'], plot_df['GDP per Working Age Adult'], alpha=0.7,
                        c=plot_df['Income Category'].map({'High Income': 'blue', 'Other': 'orange'}))
            plt.title('GDP per Capita vs GDP per Working Age Adult')
            plt.xlabel('GDP per Capita')
            plt.ylabel('GDP per Working Age Adult')
            plt.grid(True)
        else:
            print("Not enough data to create scatter plot after dropping NaNs.")
        # plt.colorbar(label='Income Category') 
        # plt.show() 
    else:
        print("Cannot create Matplotlib scatter plot due to missing columns.")


    # Aggregated Insights
    if 'Income Category' in df.columns:
        agg_cols = {
            'Average_GDP': ('GDP', 'mean'),
            'Average_GDP_per_Working_Age': ('GDP per Working Age Adult', 'mean'),
            'Average_GDP_per_Capita': ('GDP per Capita', 'mean'),
            'Population': ('Total_Population', 'sum')
        }
        # Filter out columns that don't exist in df for aggregation
        valid_agg_cols = {k: v for k, v in agg_cols.items() if v[0] in df.columns}
        
        if valid_agg_cols:
            grouped_data = df.groupby('Income Category').agg(**valid_agg_cols).reset_index()
            print("\\nAggregated Data by Income Category:")
            print(grouped_data)
        else:
            print("No valid columns found for aggregation by Income Category.")
    else:
        print("Cannot create aggregated insights because 'Income Category' column is missing.")


def create_dashboard(df_initial):
    """Creates and configures the Dash application."""
    if df_initial is None or df_initial.empty:
        print("Cannot create dashboard without data or with empty data.")
        # Return a simple app layout with an error message
        app = Dash(__name__)
        app.layout = html.Div([html.H1("Error: Data not available for dashboard.")])
        return app

    app = Dash(__name__)

    # Define available metrics for dropdown based on existing columns
    available_metrics = []
    metric_options_map = {
        'GDP per Capita': 'GDP per Capita',
        'GDP per Working Age Adult': 'GDP per Working Age Adult',
        'PPP GDP per Capita': 'PPP GDP per Capita',
        'PPP GDP per Working Age Adult': 'PPP GDP per Working Age Adult'
    }
    for label, col_name in metric_options_map.items():
        if col_name in df_initial.columns and not df_initial[col_name].isnull().all():
            available_metrics.append({'label': label, 'value': col_name})

    if not available_metrics:
        app.layout = html.Div([html.H1("Error: No valid metrics available for plotting in dashboard.")])
        return app

    app.layout = html.Div([
        html.H1('GDP and Population Analysis Dashboard', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='metric-dropdown',
            options=available_metrics,
            value=available_metrics[0]['value'] if available_metrics else None, # Default to first available
            placeholder='Select a metric to visualize'
        ),
        dcc.Graph(id='scatter-plot'),
        html.P('Select Income Category:'),
        dcc.Checklist(
            id='income-filter',
            options=[
                {'label': 'High Income', 'value': 'High Income'},
                {'label': 'Other', 'value': 'Other'}
            ] if 'Income Category' in df_initial.columns else [],
            value=['High Income', 'Other'] if 'Income Category' in df_initial.columns else [],
            inline=True
        )
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('metric-dropdown', 'value'),
        Input('income-filter', 'value')
    )
    def update_scatter(metric, income_filter_values):
        if not metric: # No metric selected or available
            return px.scatter(title="Please select a metric.")

        if 'Income Category' not in df_initial.columns or not income_filter_values:
            # If Income Category doesn't exist or no filter is selected, use all data for the selected metric
            filtered_data = df_initial.copy()
        else:
            filtered_data = df_initial[df_initial['Income Category'].isin(income_filter_values)]
        
        if filtered_data.empty or metric not in filtered_data.columns or filtered_data[metric].isnull().all():
             return px.scatter(title=f"No data for selected metric '{metric}' and filters.")

        fig = px.scatter(
            filtered_data,
            x='Total_Population' if 'Total_Population' in filtered_data.columns else None, # Handle if Total_Population is missing
            y=metric,
            color='Income Category' if 'Income Category' in filtered_data.columns else None, # Handle if Income Category is missing
            hover_name='Country Name' if 'Country Name' in filtered_data.columns else None,
            title=f'{metric} vs Total Population'
        )
        return fig

    return app

# Main execution
if __name__ == '__main__':
    config = load_config(CONFIG_FILE_PATH)
    
    if config:
        data_file = config.get('data_file_path')
        high_income_percentile = config.get('high_income_percentile_threshold', 80)
        series_mapping = config.get('series_name_mapping')
        pivot_config = {
            'pivot_index_columns': config.get('pivot_index_columns'),
            'pivot_columns_column': config.get('pivot_columns_column'),
            'pivot_values_column': config.get('pivot_values_column')
        }

        if not all([data_file, series_mapping, pivot_config['pivot_index_columns'], pivot_config['pivot_columns_column'], pivot_config['pivot_values_column']]):
            print("Error: Missing critical configuration for data loading or reshaping. Check config.json.")
        else:
            raw_data = load_data(data_file)
            
            if raw_data is not None:
                reshaped_data = reshape_data(raw_data, pivot_config, series_mapping)

                if reshaped_data is not None:
                    transformed_data = clean_and_transform_data(reshaped_data.copy(), high_income_percentile) 
                    
                    if transformed_data is not None:
                        perform_exploratory_analysis(transformed_data)
                        dashboard_app = create_dashboard(transformed_data) 
                        
                        if dashboard_app:
                            dashboard_app.run_server(debug=True)
                    else:
                        print("Data cleaning and transformation failed.")
                else:
                    print("Data reshaping failed.")
            else:
                print("Data loading failed.")
    else:
        print("Configuration loading failed. Exiting.")
