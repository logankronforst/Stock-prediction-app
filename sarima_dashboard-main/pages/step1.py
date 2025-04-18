import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash.exceptions import PreventUpdate
from datetime import datetime as dt

dash.register_page(__name__, name='1-Data set up', title='SARIMA | 1-Data set up')

from assets.fig_layout import my_figlayout, my_linelayout

_data_airp = pd.read_csv('/workspaces/CS329E/sarima_dashboard-main/data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')


def stock_price(n, start_date, end_date, val): 
    
    if n == None: 
        return [""]
    if val == None:
        raise PreventUpdate
    else: 
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else: 
            df = yf.download(val)
    df.reset_index(inplace=True)
    fig = plot_data(df)

    














### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Your dataset'])], width=12, className='row-titles')
    ]),

    # data input
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Input stock code:'], className='input-place')], width=2),
        dbc.Col([
            dbc.Input(id="dropdown_tickers", type="text", className="text-dark"),
            dbc.Button("Submit", outline=True, color = "Success")]
        , width=4),
        dbc.Col([], width = 3)
    ], className='input-place'),

    dbc.Row([
        dbc.Col([], width = 3),
        dcc.DatePickerRange(id='my-date-picker-range',
                            min_date_allowed=dt(1995, 8, 5),
                            max_date_allowed=dt.now(),
                            initial_visible_month=dt.now(),
                            end_date=dt.now().date()),
    ],
            className='date'),
    
        
        
        

    


    # raw data fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph'))
        ], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content')
    
])

### PAGE CALLBACKS ###############################################################################################################
















# Update fig
@callback(
    Output(component_id='fig-pg1', component_property='figure'),
    Input(component_id='radio-dataset', component_property='value')
)










def plot_data(df):
    

    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict()))

    fig.update_layout(title='Dataset Linechart', xaxis_title='Time', yaxis_title='Close', height = 500)
    fig.update_traces(overwrite=True, line=my_linelayout)

    return fig