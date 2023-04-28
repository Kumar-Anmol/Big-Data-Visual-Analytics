# %%
from shapely.ops import cascaded_union
from shapely.geometry import shape, mapping
import pandas as pd
import numpy as np
import json
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import ipywidgets as widgets
import plotly
import plotly.graph_objects as go

data = pd.read_csv('india_monthly_rainfall_data.csv')
# drop row which has NA
data = data.dropna()
row_to_sum = data.iloc[:, 3:15]
sum_of_row = row_to_sum.sum(axis=1)
data.insert(15, "Sum", sum_of_row, True)
ll = data['State'].unique().tolist()

data1 = data.groupby('State')
state_name = 'Andhra Pradesh'
data_sum = data1.get_group(state_name).groupby('Year').sum()['Sum']/data.groupby(
    'State').get_group(state_name).groupby('District').nunique().count()[0]  # .iloc[:,-1]/102

max_index = data_sum.idxmax()
min_index = data_sum.idxmin()
fig = px.line(data, x=data_sum.index, y=data_sum)
# fig.add_annotation(x=max_index, y=data_sum[max_index],
#                    text="Max: {:.2f}".format(data_sum[max_index]),
#                    showarrow=True, arrowhead=1)
# fig.add_annotation(x=min_index, y=data_sum[min_index],
#                    text="Min: {:.2f}".format(data_sum[min_index]),
#                    showarrow=True, arrowhead=1)
fig.update_layout(height=400, showlegend=False)
fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='Rainfall')
# update the title
fig.update_layout(title_text='Yearwise rainfall of '+state_name)


# state_name='Andhra Pradesh'
state_name1 = 'Bihar'
data1 = data.groupby('State')

data_sum = data1.get_group(state_name).groupby('Year').sum()[
    'Sum']/data.groupby('State').get_group(state_name).groupby('District').nunique().count()[0]
data_sum1 = data1.get_group(state_name1).groupby('Year').sum()[
    'Sum']/data.groupby('State').get_group(state_name1).groupby('District').nunique().count()[0]

max_index = data_sum.idxmax()
min_index = data_sum.idxmin()

fig1 = px.line(data, x=data_sum.index, y=data_sum)
# , mode='lines', name=state_name1)
fig1.add_scatter(x=data_sum1.index, y=data_sum1)
# fig = px.line(data.groupby('State').get_group('Andhra Pradesh').groupby('Year').sum()['Sum']/102,x=data.groupby('State').get_group(state_name).groupby('Year').sum()['Sum']/102.index, y=data.groupby('State').get_group(state_name).groupby('Year').sum()['Sum']/102, title='Life expectancy in Canada')
# fig.add_annotation(x=max_index, y=data_sum[max_index],
#                    text="Max: {:.2f}".format(data_sum[max_index]),
#                    showarrow=True, arrowhead=1)
# fig.add_annotation(x=min_index, y=data_sum[min_index],
#                    text="Min: {:.2f}".format(data_sum[min_index]),
#                    showarrow=True, arrowhead=1)
fig1.update_layout(height=400, showlegend=False)
fig1.update_xaxes(title_text='Year')
fig1.update_yaxes(title_text='Rainfall')
# set label for different state
fig.update_traces(marker_color='red', selector=dict(name=state_name))
# update the title
fig1.update_layout(title_text='Yearwise rainfall of ' +
                   state_name+' and '+state_name1)


data_temp = data1.get_group(state_name).sum()
np_arr = np.random.randn(13, 31)
df_arr = pd.DataFrame(np_arr)

# print unique month in Months column
month_list = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
cc = 0
for i in ll:
    data_temp = data1.get_group(i).sum()
    df_arr.iloc[0, cc] = i
    df_arr.iloc[1:, cc] = data_temp.iloc[3:15] / \
        (data.groupby('State').get_group(i).groupby(
            'District').nunique().count()[0])
    cc = cc+1

fig2 = go.Figure(data=[go.Pie(labels=df_arr.iloc[0],
                 values=df_arr.iloc[10], title='Rainfall in India')])
# fig.update_layout(height=400, showlegend=False)
# fig2.show()


year_list = data['Year'].unique().tolist()
data11 = data.groupby('Year')
total_rain = np.zeros(102)
for i in range(1901, 2003):
    total_rain[i-1901] = data11.get_group(i).sum()[3:15].sum()

max_index = total_rain.argmax()
min_index = total_rain.argmin()
fig3 = px.line(data, x=data11.sum().index, y=total_rain)
fig3.update_layout(height=400, showlegend=False)
fig3.update_xaxes(title_text='Year')
fig3.update_yaxes(title_text='Rainfall')
# update the title
fig3.update_layout(title_text='Yearwise rainfall of India')
# fig3.update_annotations(dict(xref="paper", yref="paper",
#                                 x=0.0, y=1.05,
#                                 xanchor="left", yanchor="bottom",
#                                 font_size=14,
#                                 showarrow=False))
# fig3.add_annotation(x=max_index, y=total_rain[max_index],
#                    text="Max: {:.2f}".format(total_rain[max_index]),
#                    showarrow=True, arrowhead=1)
# fig3.add_annotation(x=min_index, y=total_rain[min_index],
#                    text="Max: {:.2f}".format(total_rain[min_index]),
#                    showarrow=True, arrowhead=1)

# fig3.show()


# %%
# Read
india = json.load(open('state/india_states.geojson'))

#  Read in the district-level data
districts = gpd.read_file('district/india_district.geojson')

# # Read in the rainfall data
df = pd.read_csv('india_monthly_rainfall_data.csv')

# %%
# combine ladakh and jammu and kashmir area into one in the geojson file
for i in range(len(india['features'])):
    if india['features'][i]['properties']['ST_NM'] == 'Jammu & Kashmir':
        india['features'][i]['properties']['ST_NM'] = 'Jammu and Kashmir and Ladakh'
        jk = shape(india['features'][i]['geometry'])
    if india['features'][i]['properties']['ST_NM'] == 'Ladakh':
        india['features'][i]['properties']['ST_NM'] = 'Jammu and Kashmir and Ladakh'
        ld = shape(india['features'][i]['geometry'])

jkld = cascaded_union([jk, ld])
# remove ladakh and jammu and kashmir area from the geojson file
india['features'] = [i for i in india['features']
                     if i['properties']['ST_NM'] != 'Jammu and Kashmir and Ladakh']
# add the combined ladakh and jammu and kashmir area to the geojson file
india['features'].append({'type': 'Feature', 'properties': {
                         'ST_NM': 'Jammu & Kashmir'}, 'geometry': mapping(jkld)})

# do same for Andhra Pradesh and Telangana
for i in range(len(india['features'])):
    if india['features'][i]['properties']['ST_NM'] == 'Andhra Pradesh':
        india['features'][i]['properties']['ST_NM'] = 'Andhra Pradesh and Telangana'
        ap = shape(india['features'][i]['geometry'])
    if india['features'][i]['properties']['ST_NM'] == 'Telangana':
        india['features'][i]['properties']['ST_NM'] = 'Andhra Pradesh and Telangana'
        ts = shape(india['features'][i]['geometry'])

apts = cascaded_union([ap, ts])
india['features'] = [i for i in india['features']
                     if i['properties']['ST_NM'] != 'Andhra Pradesh and Telangana']
india['features'].append({'type': 'Feature', 'properties': {
                         'ST_NM': 'Andhra Pradesh'}, 'geometry': mapping(apts)})

# %%
# create a dictionary to map the state name to a number
state_id_map = {}
for feature in india['features']:
    feature['id'] = feature['properties']['ST_NM']
    state_id_map[feature['properties']['ST_NM']] = feature['id']

# %%
state_id_map['Andaman & Nicobar Islands'] = state_id_map['Andaman & Nicobar']
state_id_map['Chattisgarh'] = state_id_map['Chhattisgarh']
state_id_map['Orissa'] = state_id_map['Odisha']
state_id_map['Pondicherry'] = state_id_map['Puducherry']
state_id_map['Dadra & Nagar Haveli'] = state_id_map['Dadra and Nagar Haveli and Daman and Diu']
state_id_map['Daman & Diu'] = state_id_map['Dadra and Nagar Haveli and Daman and Diu']
state_id_map['Uttaranchal'] = state_id_map['Uttarakhand']

# %%
df['id'] = df['State'].apply(lambda x: state_id_map[x])

# %%


def plot_rainfall(month, year):
    # choose data for the year 2002
    df_dynamic = df[df['Year'] == year]
    # group the data by state
    df_dynamic = df_dynamic.groupby('State')
    # calculate the mean of the rainfall for that month for each state
    df_dynamic = df_dynamic[month].mean().reset_index()
    # add the id column to the dataframe
    df_dynamic['id'] = df_dynamic['State'].apply(lambda x: state_id_map[x])
    # plot the choropleth map
    fig = px.choropleth(df_dynamic, geojson=india, locations='id',
                        color=month, hover_name='State', hover_data=[month])
    fig.update_geos(fitbounds='locations', visible=False)
    # make the figure background transparent and text color white
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig


# %%
district_id_map = {}
for i in range(len(districts)):
    district_id_map[districts['NAME_2'][i]] = districts['ID_2'][i]
    # if var2 is not None: then split var2 by , or | and add to dict
    if districts['VARNAME_2'][i] is not np.nan:
        if '|' in districts['VARNAME_2'][i]:
            for var2 in districts['VARNAME_2'][i].split('|'):
                district_id_map[var2] = districts['ID_2'][i]
        elif ',' in districts['VARNAME_2'][i]:
            for var2 in districts['VARNAME_2'][i].split(','):
                # remove spaces
                var2 = var2.strip()
                district_id_map[var2] = districts['ID_2'][i]
        else:
            district_id_map[districts['VARNAME_2'][i]] = districts['ID_2'][i]

# %%
district_id_map['Andaman'] = district_id_map['Andaman Islands']
district_id_map['Nicobar'] = district_id_map['Nicobar Islands']
district_id_map['Dibang Valley'] = district_id_map['Lower Dibang Valley']
district_id_map['Dhubri'] = district_id_map['Dhuburi']
district_id_map['Golapara'] = district_id_map['Goalpara']
district_id_map['Paschim Champaran'] = district_id_map['West Champaran']
district_id_map['Rajnandgaon'] = district_id_map['Raj Nandgaon']
district_id_map['Dadra & Nagar Haveli'] = district_id_map['Dadra and Nagar Haveli']
district_id_map['New Delhi'] = district_id_map['Delhi']
district_id_map['Yamunanagar'] = district_id_map['Yamuna Nagar']
district_id_map['Lahul & Spiti'] = district_id_map['Lahul and Spiti']
district_id_map['Anantanag'] = district_id_map['Anantnag (Kashmir South)']
district_id_map['Badgam'] = district_id_map['Bagdam']
district_id_map['Baramula'] = district_id_map['Baramula (Kashmir North)']
district_id_map['Kupwara'] = district_id_map['Kupwara (Muzaffarabad)']
district_id_map['Leh'] = district_id_map['Ladakh (Leh)']
district_id_map['Garwah'] = district_id_map['Garhwa']
district_id_map['Hazaribagh'] = district_id_map['Hazaribag']
district_id_map['Kodarma'] = district_id_map['Koderma']
district_id_map['Pakaur'] = district_id_map['Pakur']
district_id_map['West Singbhum'] = district_id_map['Singhbhum West']
district_id_map['East Singbhum'] = district_id_map['Singhbhum East']
district_id_map['Bangalore'] = district_id_map['Bangalore Urban']
district_id_map['Chamarajanagar'] = district_id_map['Chamrajnagar']
district_id_map['Chikmangalur'] = district_id_map['Chikmagalur']
district_id_map['Dakshina Kannada'] = district_id_map['Dakshin Kannad']
district_id_map['Uttara Kannada'] = district_id_map['Uttar Kannand']
district_id_map['Pathanamthitta'] = district_id_map['Pattanamtitta']
district_id_map['Gadchiroli'] = district_id_map['Garhchiroli']
district_id_map['Mumbai'] = district_id_map['Greater Bombay']
district_id_map['Mumbai (Suburban)'] = district_id_map['Greater Bombay']
district_id_map['Imphal East'] = district_id_map['East Imphal']
district_id_map['Imphal West'] = district_id_map['West Imphal']
district_id_map['Aizwal'] = district_id_map['Aizawl']
district_id_map['Anugul'] = district_id_map['Angul']
district_id_map['Bargarh'] = district_id_map['Baragarh']
district_id_map['Baudh'] = district_id_map['Boudh']
district_id_map['Jagatsinghapur'] = district_id_map['Jagatsinghpur']
district_id_map['Jajapur'] = district_id_map['Jajpur']
district_id_map['Nabarangapur'] = district_id_map['Nabarangpur']
district_id_map['Sonapur'] = district_id_map['Sonepur']
district_id_map['Pondicherry'] = district_id_map['Puducherry']
district_id_map['Nawanshahr'] = district_id_map['Nawan Shehar']
district_id_map['East Sikkim'] = district_id_map['East']
district_id_map['Sikkim'] = district_id_map['West Sikkim']
district_id_map['Imphal East'] = district_id_map['East Imphal']
district_id_map['Imphal'] = district_id_map['West Imphal']
district_id_map['The Nilgiris'] = district_id_map['Nilgiris']
district_id_map['Thoothukkudi'] = district_id_map['Thoothukudi']
district_id_map['Tiruchirapalli'] = district_id_map['Tiruchchirappalli']
district_id_map['Tirunelveli'] = district_id_map['Tirunelveli Kattabo']
district_id_map['Viluppuram'] = district_id_map['Villupuram']
district_id_map['Barabanki'] = district_id_map['Bara Banki']
district_id_map['Budaun'] = district_id_map['Badaun']
district_id_map['Sant Ravidas Nagar'] = district_id_map['Sant Ravi Das Nagar']
district_id_map['Shrawasti'] = district_id_map['Shravasti']
district_id_map['Dehradun'] = district_id_map['Dehra Dun']
district_id_map['Nainital'] = district_id_map['Naini Tal']
district_id_map['Rudraprayag'] = district_id_map['Rudra Prayag']
district_id_map['Burdwan'] = district_id_map['Barddhaman']
district_id_map['Cooch Behar'] = district_id_map['Kochbihar']
district_id_map['South Dinajpur'] = district_id_map['Dakshin Dinajpur']
district_id_map['Darjeeling'] = district_id_map['Darjiling']
district_id_map['Howrah'] = district_id_map['Haora']
district_id_map['Hooghly'] = district_id_map['Hugli']
district_id_map['Malda'] = district_id_map['Maldah']
district_id_map['Midnapore'] = district_id_map['West Midnapore']
district_id_map['Purulia'] = district_id_map['Puruliya']
district_id_map['Lakshadweep'] = district_id_map['Kavaratti']
district_id_map['Goa'] = district_id_map['North Goa']

# %%
df['ID_2'] = df['District'].apply(lambda x: district_id_map[x])

# %%
state_id_map['Andaman and Nicobar'] = state_id_map['Andaman & Nicobar']
state_id_map['Dadra and Nagar Haveli'] = state_id_map['Dadra & Nagar Haveli']
state_id_map['Daman and Diu'] = state_id_map['Daman & Diu']
state_id_map['Jammu and Kashmir'] = state_id_map['Jammu & Kashmir']

# %%
districts['sid'] = districts['NAME_1'].apply(lambda x: state_id_map[x])

# %%


def state_map(month, year, state_id):
    # choose data for the year and state
    df_dynamic = df[df['Year'] == year]
    df_dynamic = df_dynamic[df_dynamic['id'] == state_id]
    # choose the map for the state
    districts_dynamic = districts[districts['sid'] == state_id]
    # name  the ID_2 column as id
    districts_dynamic = districts_dynamic.rename(columns={'ID_2': 'id'})
    # specify this id as featureid key
    districts_dynamic['id'] = districts_dynamic['id'].astype(str)
    districts_dynamic = districts_dynamic.set_index('id')
    districts_dynamic = districts_dynamic.to_json()
    districts_dynamic = json.loads(districts_dynamic)
    district_fig = px.choropleth(
        df_dynamic, geojson=districts_dynamic, locations='ID_2', color=month, hover_name='District')
    district_fig.update_geos(fitbounds='locations', visible=False)
    # make background transparent
    district_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    # make text white
    district_fig.update_layout(font_color='white')
    return district_fig


# %%
# Create the app
app = dash.Dash(__name__)

server = app.server


# %%
# Create the layout
app.layout = html.Div([
    html.H1('India District Level Data'),
    html.H2('Hover over the map to see the rainfall'),
    html.H3('Choose year'),
    dcc.Slider(
        id='year',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        # make this slider smaoother and place it in the center
        marks={str(year): str(year) for year in df['Year'].unique()},
        step=None, included=False, updatemode='drag',
        tooltip={'always_visible': True, 'placement': 'bottom'},
    ),
    html.H3('Choose month'),
    dcc.Dropdown(
        id='month',
        options=[{'label': i, 'value': i} for i in df.columns[3:15]],
        value='Jan'        # make this dropdown smaller and place it in the center
        , style={'width': '40%', 'margin': 'auto', 'border': '2px solid black'}
    ),
    html.Div([
        html.H3('India Map'),
        html.H4('Click on the state to see the district map'),
        dcc.Graph(id='india_map', figure=plot_rainfall(
            'Jan', 2002), style={'border': '2px solid black'}),
        html.H3('State Map'),
        html.H4('Move cursor over the district to see the rainfall'),
        dcc.Graph(id='district_map', figure=state_map(
            'Jan', 2002, 'Bihar'), style={'border': '2px solid black'})
    ],
        # display the india map and state map side by side
        style={'columnCount': 2}),

    html.Div([
        html.H1('SUMMARY', style={'text-align': 'center'})
    ]),
    html.Div([
        # add label to centre
        html.H4('Yearly Rainfall', style={'text-align': 'center'}),
        dcc.Dropdown(
            id='first-dropdown',
            value='Andhra Pradesh',
            options=ll,
            style={'width': '300px', 'height': '40px',
                   'margin': 'auto', 'border': '2px solid black'}
            # style={'border': '2px solid black'}

        ),
        dcc.Graph(
            figure=fig,
            id='example',
            style={'border': '2px solid black'}
        ),
    ],
    ),
    html.Div([
        html.H4('Yearly Rainfall Comparision', style={'text-align': 'center'}),
        dcc.Dropdown(
            id='second-dropdown',
            value='Andhra Pradesh',
            options=ll,
            style={'width': '300px', 'height': '40px', 'margin': 'auto', 'border': '2px solid black'}),
        dcc.Dropdown(
            id='third-dropdown',
            value='Bihar',
            options=ll,
            style={'width': '300px', 'height': '40px', 'margin': 'auto', 'border': '2px solid black'}),
        dcc.Graph(
            figure=fig1,
            id='example1',
            style={'border': '2px solid black'}
        )
    ]),
    html.Div([
        html.H4('Month-wise Average Rainfall', style={'text-align': 'center'}),
        dcc.Dropdown(
            id='fourth-dropdown',
            value='January',
            options=month_list,
            style={'width': '300px', 'height': '40px', 'margin': 'auto', 'border': '2px solid black'}),
        # style={'border': '2px solid black'}),
        dcc.Graph(

            figure=fig2,
            id='example2',
            style={'border': '2px solid black'}
        )
    ]),
    html.Div([
        # add label to centre
        html.H4('Overall Yearwise Rainfall in India',
                style={'text-align': 'center'}),

        dcc.Graph(
            figure=fig3,
            id='example3',
            style={'border': '2px solid black'}
        ),
    ],
    ),
], style={'textAlign': 'center', 'color': '#000000', 'backgroundColor': 'rgb(105, 145, 181)'})

# Create the callback


@app.callback(
    Output('district_map', 'figure'),
    [Input('month', 'value'), Input('year', 'value'),
     Input('india_map', 'clickData')]
)
def update_map(month, year, state_id):
    if state_id is None:
        state_id = 'Bihar'
    else:
        state_id = state_id['points'][0]['location']
    return state_map(month, year, state_id)

# callback for the india map


@app.callback(
    Output('india_map', 'figure'),
    [Input('month', 'value'), Input('year', 'value')]
)
def update_map(month, year):
    return plot_rainfall(month, year)


@app.callback(
    dash.dependencies.Output('example', 'figure'),
    [dash.dependencies.Input('first-dropdown', 'value')])
def update_figure(selected_state):
    data_sum = data1.get_group(selected_state).groupby('Year').sum()['Sum']/data.groupby(
        'State').get_group(selected_state).groupby('District').nunique().count()[0]
    max_index = data_sum.idxmax()
    min_index = data_sum.idxmin()
    fig = px.line(data, x=data_sum.index, y=data_sum)
    fig.add_annotation(x=max_index, y=data_sum[max_index],
                       text="Max: {:.2f}".format(data_sum[max_index]),
                       showarrow=True, arrowhead=1)
    fig.add_annotation(x=min_index, y=data_sum[min_index],
                       text="Min: {:.2f}".format(data_sum[min_index]),
                       showarrow=True, arrowhead=1)
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Rainfall')
    # update the title
    fig.update_layout(title_text='Yearwise rainfall of '+selected_state)
    return fig


@app.callback(
    dash.dependencies.Output('example1', 'figure'),
    [dash.dependencies.Input('second-dropdown', 'value'),
        dash.dependencies.Input('third-dropdown', 'value')])
def update_figure(selected_state, selected_state1):
    data_sum = data1.get_group(selected_state).groupby('Year').sum()['Sum']/data.groupby(
        'State').get_group(selected_state).groupby('District').nunique().count()[0]
    data_sum1 = data1.get_group(selected_state1).groupby('Year').sum()['Sum']/data.groupby(
        'State').get_group(selected_state1).groupby('District').nunique().count()[0]
    max_index = data_sum.idxmax()
    min_index = data_sum.idxmin()
    fig = px.line(data, x=data_sum.index, y=data_sum)
    fig.add_scatter(x=data_sum1.index, y=data_sum1)
    fig.update_layout(height=400, showlegend=True, legend=dict(
        traceorder="grouped", title_text=selected_state1))
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Rainfall')
    # update label for different state
    fig.update_traces(marker_color='red', selector=dict(name=selected_state1))

    # update the title
    fig.update_layout(title_text='Yearwise rainfall of ' +
                      selected_state+' and '+selected_state1)
    return fig


@app.callback(
    dash.dependencies.Output('example2', 'figure'),
    [dash.dependencies.Input('fourth-dropdown', 'value')])
def update_figure(select_month):
    temp = 0
    if (select_month == 'January'):
        temp = 1
    elif (select_month == 'February'):
        temp = 2
    elif (select_month == 'March'):
        temp = 3
    elif (select_month == 'April'):
        temp = 4
    elif (select_month == 'May'):
        temp = 5
    elif (select_month == 'June'):
        temp = 6
    elif (select_month == 'July'):
        temp = 7
    elif (select_month == 'August'):
        temp = 8
    elif (select_month == 'September'):
        temp = 9
    elif (select_month == 'October'):
        temp = 10
    elif (select_month == 'November'):
        temp = 11
    elif (select_month == 'December'):
        temp = 12

    fig = go.Figure(data=[go.Pie(labels=df_arr.iloc[0], values=df_arr.iloc[temp],
                    title='Rainfall in' + select_month + ' in different states')])
    fig.update_layout(height=600, showlegend=True)
    return fig


# %%
# Run the app
if __name__ == '__main__':
    app.run_server(port=80)
