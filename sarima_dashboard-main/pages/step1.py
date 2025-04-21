import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from dash.exceptions import PreventUpdate
from datetime import datetime as dt

dash.register_page(__name__, name='1-Input Stock Code', title='Stock Prediction | 1-Input Stock Code')

from assets.fig_layout import my_figlayout, my_linelayout




### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3()], width=12, className='row-titles')
    ]),

    # data input
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([html.P(['Input stock code:'], className='input-place')], width=2),
        dbc.Col([
            dbc.Input(id="dropdown_tickers", type="text", className="text-dark"),
            dbc.Button("Submit",id='submit',outline=True, color = "Success")]
        , width=1),
        dbc.Col([], width = 2)
        
    ], className='input-place'),

        dbc.Col([]),   
            dcc.Slider(0, 24, 
                step=None,
                marks={
                    1 : '1M',
                    3 : '3M',
                    6 : '6M',
                    12 : '1Y',
                    24 : '2Y'
                },            
                value=5
            ),
        dbc.Col([]),
        
        
    
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
    Input("submit", "n_clicks"),
    State("dropdown_tickers", "value"),
)


def stock_price(n_clicks, ticker): 
    
    if not n_clicks or not ticker:
        raise PreventUpdate
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01')
    
    df.reset_index(inplace=True)

    return plot_data(df)


def plot_data(df):
    

    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close']))

    fig.update_layout(title='Dataset Linechart', xaxis_title='Time', yaxis_title='Close', height = 500)
    # fig.update_traces(overwrite=True, line=my_linelayout)

    return fig