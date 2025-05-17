import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Load and inspect the data
file_path = 'C:/Users/aaron/OneDrive/Desktop/Sandboxes/GDP_and_Population_by_country_since_1960\Data_GDP_Pop_by_Country_1960_Countries_only.csv'  # Replace with the uploaded file path
data = pd.read_csv(file_path)
data.head()

# Data Cleaning and Transformation
# Replace missing or placeholder values with NaN
data.replace(['..', 'N/A', ''], np.nan, inplace=True)

# Convert relevant columns to numeric where possible
for col in ['GDP', 'PPP_GDP', 'Working_Age_Population', 'Total_Population']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Calculate additional metrics
data['GDP per Capita'] = data['GDP'] / data['Total_Population']
data['GDP per Working Age Adult'] = data['GDP'] / data['Working_Age_Population']
data['PPP GDP per Capita'] = data['PPP_GDP'] / data['Total_Population']
data['PPP GDP per Working Age Adult'] = data['PPP_GDP'] / data['Working_Age_Population']

# Define high-income countries (top 20% by GDP per capita)
threshold = np.percentile(data['GDP per Capita'].dropna(), 80)
data['Income Category'] = np.where(data['GDP per Capita'] >= threshold, 'High Income', 'Other')

# Exploratory Analysis
# Correlation between GDP per working age adult and total GDP
correlation = data[['GDP', 'GDP per Working Age Adult']].corr().iloc[0, 1]
print(f"Correlation between GDP and GDP per Working Age Adult: {correlation}")

# Plot GDP per Capita vs GDP per Working Age Adult
plt.figure(figsize=(10, 6))
plt.scatter(data['GDP per Capita'], data['GDP per Working Age Adult'], alpha=0.7, c=data['Income Category'].map({'High Income': 'blue', 'Other': 'orange'}))
plt.title('GDP per Capita vs GDP per Working Age Adult')
plt.xlabel('GDP per Capita')
plt.ylabel('GDP per Working Age Adult')
plt.grid(True)
plt.colorbar(label='Income Category')
plt.show()

# Create Aggregated Insights
grouped_data = data.groupby('Income Category').agg(
    Average_GDP=('GDP', 'mean'),
    Average_GDP_per_Working_Age=('GDP per Working Age Adult', 'mean'),
    Average_GDP_per_Capita=('GDP per Capita', 'mean'),
    Population=('Total_Population', 'sum')
).reset_index()
print(grouped_data)

# Build a Dash App for Interactive Analysis
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
        value=['High Income', 'Other'],
        inline=True
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('income-filter', 'value')
)
def update_scatter(metric, income_filter):
    filtered_data = data[data['Income Category'].isin(income_filter)]
    fig = px.scatter(
        filtered_data,
        x='Total_Population',
        y=metric,
        color='Income Category',
        hover_name='Country Name',
        title=f'{metric} vs Total Population'
    )
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
