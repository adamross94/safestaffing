import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Read the CSV data
df = pd.read_csv(
    r'data/Safe_Staffing_Data.csv')

# Define the list of wards needed for Safe Staffing
wards_of_interest = [
    "N - General HDU", "N - ICU", "N - Kent Ward", "N - Kingfisher Ward",
    "N - Lawrence Ward", "N - Medical HDU", "N - Pearl Ward", "N - Pembroke Ward",
    "N - Phoenix Ward", "N - SDCC", "N - The Birth Place", "N - Victory Ward",
    "N - Emerald AAU", "N - Emerald Ward", "N - Harvey Ward", "N - Jade Ward",
    "N - Keats Ward", "N - Lister Assessment Unit", "N - McCulloch Ward",
    "N - Nelson Ward", "N - Sapphire Ward", "N - Tennyson Ward", "N - Will Adams Ward",
    "N - Wakeley", "N - Ocelot", "N - Nelson", "N - Wakeley Ward", "N - Jade Escalation",
    "N - Nelson Esc", "N - Wakeley Esc", "N - Sheppey Frailty Unit", "N - Harvey"
]

# Filter the DataFrame to include only the rows with the specified wards
df = df[df['Name'].isin(wards_of_interest)]

# Assuming 'Month' column contains dates in 'DD/MM/YYYY' format
# Convert 'Month' column to datetime
df['Month'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')

# Extract year
df['Year'] = df['Month'].dt.year

# Calculate financial quarter from the month
df['Quarter'] = df['Month'].dt.month.apply(lambda x: f"Q{((x - 1) // 3) + 1}")

# Format sorted months for dropdown display
month_options = [{'label': month.strftime('%b %Y'), 'value': month.strftime(
    '%Y-%m')} for month in sorted(df['Month'].dt.to_period('M').unique().to_timestamp(), reverse=True)]

# Now, you can use these 'Year' and 'Quarter' columns to populate your dropdowns
# Ensure years are unique and sorted
unique_years = sorted(df['Year'].unique())

# Year Dropdown options
year_options = [{'label': str(year), 'value': year} for year in unique_years]

# Since quarters are consistent across years, you don't need to extract unique values from the DataFrame
quarter_options = [{'label': quarter, 'value': quarter}
                   for quarter in ['Q1', 'Q2', 'Q3', 'Q4']]

# Sort months in reverse order, filtering out any None values
sorted_months = sorted(df[df['Month'].notnull()]
                       ['Month'].unique(), reverse=True)

hour_columns = [col for col in df.columns if 'Hrs' in col]

# Clean and preprocess the data
# Convert hour columns to numeric and round to 0 decimal places
for col in hour_columns:
    # Convert to numeric, setting errors='coerce' will turn non-convertible values into NaN
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

    # Round the numbers
    df[col] = df[col].round(0)

    # Fill NaN values with 0
    df[col] = df[col].fillna(0)

# Insert the new column calculations here
df['Day Average Fill Rate - Reg'] = df['Day Reg Actual Hrs'] / \
    df['Day Reg Planned Hrs']
df['Day Average Fill Rate - Reg'] = df['Day Average Fill Rate - Reg'].fillna(0)
df['Day Average Fill Rate - Unreg'] = df['Day Unreg Actual Hrs'] / \
    df['Day Unreg Planned Hrs']
df['Day Average Fill Rate - Unreg'] = df['Day Average Fill Rate - Unreg'].fillna(
    0)
df['Night Average Fill Rate - Reg'] = df['Night Reg Actual Hrs'] / \
    df['Night Reg Planned Hrs']
df['Night Average Fill Rate - Reg'] = df['Night Average Fill Rate - Reg'].fillna(
    0)
df['Night Average Fill Rate - Unreg'] = df['Night Unreg Actual Hrs'] / \
    df['Night Unreg Planned Hrs']
df['Night Average Fill Rate - Unreg'] = df['Night Average Fill Rate - Unreg'].fillna(
    0)
df['Patient Count At Midnight'] = df['Patient Count At Midnight'].replace(
    0, np.nan)
df['CHPPD Reg Staff'] = (df['Day Reg Actual Hrs'] +
                         df['Night Reg Actual Hrs']) / df['Patient Count At Midnight']
df['CHPPD Unreg Staff'] = (df['Day Unreg Actual Hrs'] +
                           df['Night Unreg Actual Hrs']) / df['Patient Count At Midnight']
df['CHPPD Overall'] = (df['Day Reg Actual Hrs'] + df['Day Unreg Actual Hrs'] +
                       df['Night Reg Actual Hrs'] + df['Night Unreg Actual Hrs']) / df['Patient Count At Midnight']
df['CHPPD Overall'] = df['CHPPD Overall'].fillna(0)

# Calculate simplified CHPPD values for the DataFrame, handling NaN values in the mean calculation
chppd_reg_simple = df['CHPPD Reg Staff'].mean(skipna=True)
chppd_unreg_simple = df['CHPPD Unreg Staff'].mean(skipna=True)

# Select columns E to L by their integer location (4 to 11, as indexing starts at 0)
# and check if all values in these columns are 0 for each row
mask = (df.iloc[:, 4:12] == 0).all(axis=1)

# Invert the mask to keep rows where not all values in columns E to L are 0
df_filtered = df[~mask]

# Create a mapping of ward names to specialties
specialty_mapping = {
    "N - Lawrence Ward": "823 - HAEMATOLOGY - STANDARD",
    "N - Emerald AAU": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Emerald Ward": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Jade Ward": "502 - GYNAECOLOGY - STANDARD",
    "N - Keats Ward": "301 - GASTROENTEROLOGY - STANDARD",
    "N - Lister Assessment Unit": "100 - GENERAL SURGERY - STANDARD",
    "N - McCulloch Ward": "100 - GENERAL SURGERY - STANDARD",
    "N - Nelson Ward": "320 - CARDIOLOGY - STANDARD",
    "N - Sapphire Ward": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Sheppey Frailty Unit": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Tennyson Ward": "430 - GERIATRIC MEDICINE - STANDARD",
    "N - Wakeley Esc": "300 - GENERAL MEDICINE - STANDARD",
    "N - Will Adams Ward": "301 - GASTROENTEROLOGY - STANDARD",
    "N - General HDU": "192 - CRITICAL CARE MEDICINE - STANDARD",
    "N - Harvey Ward": "328 - STROKE MEDICINE - STANDARD",
    "N - ICU": "192 - CRITICAL CARE MEDICINE - STANDARD",
    "N - Kingfisher Ward": "100 - GENERAL SURGERY - STANDARD",
    "N - Ocelot": "110 - TRAUMA & ORTHOPAEDICS - STANDARD",
    "N - Pembroke Ward": "110 - TRAUMA & ORTHOPAEDICS - STANDARD",
    "N - Phoenix Ward": "100 - GENERAL SURGERY - STANDARD",
    "N - SDCC": "100 - GENERAL SURGERY - STANDARD",
    "N - Victory Ward": "100 - GENERAL SURGERY - STANDARD",
    "N - Kent Ward": "501 - OBSTETRICS - STANDARD",
    "N - Pearl Ward": "501 - OBSTETRICS - STANDARD",
    "N - The Birth Place": "501 - OBSTETRICS - STANDARD"
}

# Map specialties to wards and drop any rows where the specialty is None
df['Specialty'] = df['Name'].map(specialty_mapping).dropna()


# Generate specialty options, ensuring no None values and removing duplicates
# Drop any None values and get unique specialties
specialties = df['Specialty'].dropna().unique()
filtered_specialty_options = [
    {'label': specialty, 'value': specialty} for specialty in specialties]

# Remove any duplicates from the filtered_specialty_options list
# Convert each dictionary to a tuple (to make it hashable), create a set (to remove duplicates), and then convert back to a dictionary
filtered_specialty_options = [dict(t) for t in {tuple(
    d.items()) for d in filtered_specialty_options}]

# Initialize the app
app = dash.Dash(__name__)
server = app.server

# Define the layout with a global style for font
app.layout = html.Div([
    # Header
    html.Header([
        html.Img(src='/assets/images/bi_visual_identity.png',
                 style={'height': '150px', 'display': 'inline-block'})
    ], style={'background-color': '#003087', 'padding': '10px', 'color': 'white', 'textAlign': 'left'}),

    # Main content and footer wrapper
    html.Div([
        # Main content with side menu and graph grid
        html.Div([
            # Side menu
            html.Div([
                html.H1('Safe Staffing Report', style={
                        'text-align': 'center', 'margin-bottom': '10px'}),
                html.P('''
                        This interactive dashboard is designed to provide comprehensive insights into staffing levels within various hospital wards, specialties, and divisions, ensuring optimal patient care and operational efficiency. By leveraging real-time data, the dashboard allows for the monitoring of registered and unregistered staff fill rates, care hours per patient day (CHPPD), and staffing distribution across different shifts.
                        ''',
                        style={'text-align': 'center', 'margin-bottom': '20px'}),
                html.H2('Options', style={'text-align': 'center'}),
                dcc.Dropdown(
                    id='name-dropdown',
                    options=[{'label': name, 'value': name}
                             for name in df['Name'].unique()],
                    placeholder='Select Name',
                    style={'marginBottom': '20px'}
                ),
                # Specialty Dropdown
                dcc.Dropdown(
                    id='specialty-dropdown',
                    options=filtered_specialty_options,
                    placeholder='Select Specialty',
                    style={'marginBottom': '20px'}
                ),
                dcc.Dropdown(
                    id='division-dropdown',
                    options=[{'label': division, 'value': division}
                             for division in df['Division'].unique()],
                    placeholder='Select Division',
                    style={'marginBottom': '20px'}
                ),
                # Month Dropdown, sorted with most recent month first
                dcc.Dropdown(
                    id='month-dropdown',
                    options=month_options,
                    placeholder='Select Month',
                    style={'marginBottom': '20px'}
                ),
                # Year Dropdown
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': year, 'value': year}
                             for year in sorted(df['Year'].unique(), reverse=True)],
                    placeholder='Select Year',
                    style={'marginBottom': '20px'}
                ),
                # Quarter Dropdown
                dcc.Dropdown(
                    id='quarter-dropdown',
                    options=[{'label': quarter, 'value': quarter}
                             for quarter in sorted(df['Quarter'].unique())],
                    placeholder='Select Quarter',
                    style={'marginBottom': '20px'}
                )
            ], style={'width': '25%', 'padding': '20px', 'background-color': '#E8EDEE'}),

            # Graphs
            html.Div([
                html.Div([dcc.Graph(id='graph-1')], style={'width': '50%'}),
                html.Div([dcc.Graph(id='graph-2')], style={'width': '50%'}),
                html.Div([dcc.Graph(id='graph-3')], style={'width': '50%'}),
                html.Div([dcc.Graph(id='graph-4')], style={'width': '50%'}),
                html.Div([dcc.Graph(id='graph-5')], style={'width': '50%'}),
                html.Div([dcc.Graph(id='graph-6')], style={'width': '50%'})
            ], style={'width': '75%', 'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'marginLeft': '10px', 'marginTop': '10px'})
        ], style={'display': 'flex'}),

        # Footer
        html.Footer([
            html.Img(
                # Update the src attribute to point to your image's location
                src='/assets/images/patient_first_badge.png',
                # Adjust styling to pan the image to the left
                style={'height': '60px',
                       'margin-right': 'auto', 'margin-left': '5px'}
            ),
            html.Img(src='/assets/images/nhs_england_identity.png',
                     style={'height': '60px', 'margin-left': 'auto', 'margin-right': '5px'})
        ], style={'background-color': '#003087', 'padding': '10px', 'color': 'white', 'display': 'flex', 'justify-content': 'space-between'})
    ], style={'display': 'flex', 'flex-direction': 'column', 'height': 'calc(100vh - 160px)'})
], style={'font-family': 'Calibri', 'font-size': '16px'})


# Callbacks for updating graphs
@app.callback(
    [Output(f'graph-{i}', 'figure') for i in range(1, 7)],
    [Input('name-dropdown', 'value'),
     Input('division-dropdown', 'value'),
     Input('month-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('quarter-dropdown', 'value'),
     Input('specialty-dropdown', 'value')]
)
def update_graphs(name, division, month, year, quarter, specialty):
    filtered_df = df.copy()

    # Apply filters based on dropdown selections
    if name:
        filtered_df = filtered_df[filtered_df['Name'] == name]
    if division:
        filtered_df = filtered_df[filtered_df['Division'] == division]
    if month:
        filtered_month = pd.to_datetime(month)
        filtered_df = filtered_df[filtered_df['Month'].dt.month ==
                                  filtered_month.month]
        filtered_df = filtered_df[filtered_df['Month'].dt.year ==
                                  filtered_month.year]
    if year:
        filtered_df = filtered_df[filtered_df['Year'] == int(year)]
    if quarter:
        filtered_df = filtered_df[filtered_df['Quarter'] == quarter]
    if specialty:
        filtered_df = filtered_df[filtered_df['Specialty'] == specialty]

    # Define the color sequence
    color_sequence = ["#79B927", "#37B5E5", "#8F7EA8", "#176FB7"]

    # Planned vs Actual Hours Bar Chart
    planned_columns = ['Day Reg Planned Hrs', 'Day Unreg Planned Hrs',
                       'Night Reg Planned Hrs', 'Night Unreg Planned Hrs']
    actual_columns = ['Day Reg Actual Hrs', 'Day Unreg Actual Hrs',
                      'Night Reg Actual Hrs', 'Night Unreg Actual Hrs']
    staffing_data = filtered_df[planned_columns +
                                actual_columns].sum().reset_index()
    staffing_data.columns = ['Type', 'Hours']
    staffing_data['Shift'] = staffing_data['Type'].str.extract('(Day|Night)')

    # Map boolean values in 'Reg' to 'Registered' and 'Unregistered'
    staffing_data['Reg'] = staffing_data['Type'].str.contains(
        'Reg').map({True: 'Registered', False: 'Unregistered'})

    staffing_data['Status'] = [
        'Planned' if 'Planned' in val else 'Actual' for val in staffing_data['Type']]
    fig1 = px.bar(staffing_data, x='Shift', y='Hours', color='Status',
                  barmode='group', facet_col='Reg', title='Planned vs. Actual Staff Hours',
                  color_discrete_sequence=color_sequence)

    # Iterate over the existing annotations and update the text
    for annotation in fig1.layout.annotations:
        if annotation.text.startswith('Reg='):
            # Replace 'Reg=Registered' with 'Registered' and 'Reg=Unregistered' with 'Unregistered'
            annotation.text = annotation.text.replace('Reg=', '')

    # Horizontal Bar Chart for Day vs Night Shift Hours
    # Aggregate total hours for day and night shifts
    total_day_hours = filtered_df[[
        'Day Reg Actual Hrs', 'Day Unreg Actual Hrs']].sum().sum()
    total_night_hours = filtered_df[[
        'Night Reg Actual Hrs', 'Night Unreg Actual Hrs']].sum().sum()

    # Data for the horizontal bar chart
    shift_hours = {
        'Shift Type': ['Day Shifts', 'Night Shifts'],
        'Total Hours': [total_day_hours, total_night_hours]
    }
    shift_hours_df = pd.DataFrame(shift_hours)

    # Map for assigning specific colors to day and night shifts
    color_map = {
        'Day Shifts': '#37B5E5',  # Blue for Day Shifts
        'Night Shifts': '#176FB7'  # Darker Blue for Night Shifts
    }

    # Create the horizontal bar chart with custom colors
    fig2 = px.bar(shift_hours_df, y='Shift Type', x='Total Hours', orientation='h',
                  title='Day vs. Night Shifts Comparison', color='Shift Type',
                  color_discrete_map=color_map)

    # Staff Composition Pie Chart
    # Count of 'Reg' and 'Unreg' hours
    composition_data = filtered_df[planned_columns + actual_columns].sum()
    reg_hours = composition_data.filter(like='Reg').sum()
    unreg_hours = composition_data.filter(like='Unreg').sum()
    composition_data = pd.DataFrame(
        {'Type': ['Registered', 'Unregistered'], 'Hours': [reg_hours, unreg_hours]})
    # Define the color sequence for the pie chart
    custom_color_sequence = ["#8F7EA8", "#176FB7"]
    fig3 = px.pie(composition_data, names='Type',
                  values='Hours', title='Staff Composition',
                  color_discrete_sequence=custom_color_sequence)

   # Ensure 'Month' column is in datetime format if it's not already
    filtered_df['Month'] = pd.to_datetime(
        filtered_df['Month'], format='%d/%m/%Y')

    # Sort your DataFrame by the 'Month' column to ensure chronological order
    filtered_df = filtered_df.sort_values('Month')

    # Create a new column for formatted month strings after sorting
    filtered_df['MonthFormatted'] = filtered_df['Month'].dt.strftime('%b %Y')

    # Aggregate Day and Night shift hours into new columns
    filtered_df['Day Actual Hrs'] = filtered_df[[
        'Day Reg Actual Hrs', 'Day Unreg Actual Hrs']].sum(axis=1)
    filtered_df['Night Actual Hrs'] = filtered_df[[
        'Night Reg Actual Hrs', 'Night Unreg Actual Hrs']].sum(axis=1)

    # Melt your DataFrame for the heatmap, using 'MonthFormatted'
    heatmap_data = filtered_df.melt(id_vars=['MonthFormatted'], value_vars=[
                                    'Day Actual Hrs', 'Night Actual Hrs'], var_name='Shift', value_name='Hours')

    # Ensure the melted DataFrame is sorted by 'MonthFormatted' and 'Shift' to maintain the correct order
    heatmap_data = heatmap_data.sort_values(by=['MonthFormatted', 'Shift'])

    # Pivot the melted DataFrame to create the heatmap data
    pivot_heatmap_data = heatmap_data.pivot_table(
        index='Shift', columns='MonthFormatted', values='Hours', aggfunc='sum')

    # Define a continuous custom color scale
    custom_color_scale = [
        [0, "#37B5E5"],    # Light blue for low values
        [0.5, "#E8EDEE"],  # Light grey as a neutral mid-point
        [1, "#8F7EA8"]     # Muted purple for high values
    ]

    # Plot the heatmap
    fig4 = px.imshow(pivot_heatmap_data,
                     labels=dict(x="Month", y="Shift", color="Hours"),
                     title='Staffing Distribution Across the Month',
                     color_continuous_scale=custom_color_scale)

    # Update the x-axis to reflect the sorted order of 'MonthFormatted'
    fig4.update_xaxes(categoryorder='array',
                      categoryarray=filtered_df['MonthFormatted'].unique())

    # Fill Rate Comparison Bar Chart
    fill_rate_data = {
        'Category': ['Day Reg', 'Day Unreg', 'Night Reg', 'Night Unreg'],
        'Fill Rate': [
            np.round(np.where(filtered_df['Day Reg Planned Hrs'].mean() != 0, filtered_df['Day Reg Actual Hrs'].mean(
            ) / filtered_df['Day Reg Planned Hrs'].mean() * 100, np.nan)),
            np.round(np.where(filtered_df['Day Unreg Planned Hrs'].mean() != 0, filtered_df['Day Unreg Actual Hrs'].mean(
            ) / filtered_df['Day Unreg Planned Hrs'].mean() * 100, np.nan)),
            np.round(np.where(filtered_df['Night Reg Planned Hrs'].mean() != 0, filtered_df['Night Reg Actual Hrs'].mean(
            ) / filtered_df['Night Reg Planned Hrs'].mean() * 100, np.nan)),
            np.round(np.where(filtered_df['Night Unreg Planned Hrs'].mean() != 0, filtered_df['Night Unreg Actual Hrs'].mean(
            ) / filtered_df['Night Unreg Planned Hrs'].mean() * 100, np.nan))
        ]
    }
    fig5 = px.bar(fill_rate_data, x='Category', y='Fill Rate', title='Fill Rate Comparison',
                  color='Category',
                  color_discrete_sequence=color_sequence)

    # CHPPD Chart
    # Ensure 'Patient Count At Midnight' is not zero to avoid division by zero errors
    filtered_df = filtered_df[filtered_df['Patient Count At Midnight'] > 0]

    # Calculate CHPPD for the filtered data
    filtered_df['CHPPD Reg'] = (filtered_df['Day Reg Actual Hrs'] +
                                filtered_df['Night Reg Actual Hrs']) / filtered_df['Patient Count At Midnight']
    filtered_df['CHPPD Unreg'] = (filtered_df['Day Unreg Actual Hrs'] +
                                  filtered_df['Night Unreg Actual Hrs']) / filtered_df['Patient Count At Midnight']
    # Instead of calculating CHPPD Overall as a sum, calculate it independently to ensure accurate representation
    filtered_df['CHPPD Overall'] = (filtered_df['Day Reg Actual Hrs'] + filtered_df['Night Reg Actual Hrs'] +
                                    filtered_df['Day Unreg Actual Hrs'] + filtered_df['Night Unreg Actual Hrs']) / filtered_df['Patient Count At Midnight']

    # Determine the grouping category based on the user selection
    group_category = 'Name' if name else 'Specialty' if specialty else 'Division' if division else 'Name'

    # Group the data and calculate the mean for each CHPPD type
    chppd_grouped = filtered_df.groupby(group_category).agg({
        'CHPPD Reg': 'mean',
        'CHPPD Unreg': 'mean',
        'CHPPD Overall': 'mean'
    }).reset_index()

    # Define the color sequence
    color_sequence = ["#79B927", "#37B5E5", "#8F7EA8", "#176FB7"]

    # Melt the grouped DataFrame to convert it into a long format
    chppd_melted = chppd_grouped.melt(id_vars=group_category, value_vars=['CHPPD Reg', 'CHPPD Unreg', 'CHPPD Overall'],
                                      var_name='CHPPD Type', value_name='CHPPD Value')

    # Create the grouped bar chart
    fig6 = px.bar(chppd_melted, x=group_category, y='CHPPD Value', color='CHPPD Type', barmode='group',
                  title='CHPPD by Staff Type', labels={'CHPPD Value': 'CHPPD', 'CHPPD Type': 'Staff Type'},
                  # Ensuring consistent order
                  category_orders={"CHPPD Type": [
                      "CHPPD Reg", "CHPPD Unreg", "CHPPD Overall"]},
                  color_discrete_sequence=color_sequence)

    # Determine if the filter is by ward, and adjust legend position accordingly
    legend_y_position = -1.0 if group_category == 'Name' else -0.2

    # Update layouts if necessary (e.g., layout updates, axis titles)
    fig1.update_layout(
        autosize=True,
        legend=dict(orientation='h', yanchor="top",
                    y=-0.2, xanchor="center", x=0.5)
    )
    fig2.update_layout(
        autosize=True,
        legend=dict(orientation='h', yanchor="top",
                    y=-0.2, xanchor="center", x=0.5)
    )
    fig3.update_layout(
        autosize=True,
        legend=dict(orientation='h', yanchor="top",
                    y=-0.2, xanchor="center", x=0.5)
    )
    fig4.update_layout(xaxis=dict(tickmode='array', tickvals=np.arange(len(
        filtered_df['MonthFormatted'].unique())), ticktext=filtered_df['MonthFormatted'].unique()))
    fig5.update_layout(
        autosize=True,
        legend=dict(orientation='h', yanchor="top",
                    y=-0.2, xanchor="center", x=0.5)
    )
    fig6.update_layout(
        autosize=True,
        legend=dict(orientation='h', yanchor="top",
                    y=legend_y_position, xanchor="center", x=0.5)
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
