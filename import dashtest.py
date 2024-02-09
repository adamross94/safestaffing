import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

# Sample DataFrame creation
data = {
    'Name': ['N - Lawrence Ward', 'N - Emerald AAU', 'N - Emerald Ward', 'N - Jade Ward', 'N - Keats Ward'],
    'Specialty': [None, None, None, None, None]  # Placeholder for mapping
}

df = pd.DataFrame(data)

# Create a mapping of ward names to specialties
specialty_mapping = {
    "N - Lawrence Ward": "823 - HAEMATOLOGY - STANDARD",
    "N - Emerald AAU": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Emerald Ward": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Jade Ward": "502 - GYNAECOLOGY - STANDARD",
    "N - Keats Ward": "301 - GASTROENTEROLOGY - STANDARD"
}

# Map specialties to wards
df['Specialty'] = df['Name'].map(specialty_mapping)

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='name-dropdown',
        options=[{'label': name, 'value': name}
                 for name in df['Name'].unique()],
        placeholder='Select Name',
        style={'marginBottom': '20px'}
    ),
    dcc.Dropdown(
        id='specialty-dropdown',
        placeholder='Select Specialty',
        style={'marginBottom': '20px'}
    )
])

# Callback to update the specialty dropdown options based on the selected name


@app.callback(
    Output('specialty-dropdown', 'options'),
    [Input('name-dropdown', 'value')]
)
def update_specialty_dropdown(selected_name):
    if selected_name:
        # Get the specialty for the selected name
        specialty = df[df['Name'] == selected_name]['Specialty'].iloc[0]
        if specialty:
            # Return options for the specialty dropdown
            return [{'label': specialty, 'value': specialty}]
    # If no name is selected or specialty is not available, return an empty list
    return []


if __name__ == '__main__':
    app.run_server(debug=True)
