import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import plotly.express as px

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
        dbc.Col([], width=2),
        dbc.Col([html.P(['Input stock code:'], className='input-place')], width=2),
        dbc.Col([
            dbc.Input(id="dropdown_tickers", type="text", value="TSLA", debounce=True, className="text-dark"),
        ], width=1),
        dbc.Col([], width=2)
    ], className='input-place'),

    # slider row
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.Div(
                dcc.Slider(
                    0, 24,
                    step=None,
                    marks={
                        1: '1M',
                        3: '3M',
                        6: '6M',
                        12: '1Y',
                        24: '2Y'
                    },
                    value=5,
                    id='date-slider'
                ),
                style={
                    "width": "100%",
                    "margin": "10px auto"
                }
            )
        ], width=6),
        dbc.Col([], width=2)
    ], justify="center"),

    # raw data fig
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph'))
        ], width=8),
        dbc.Col([], width=2)
    ], className='row-content')
])


### PAGE CALLBACKS ###############################################################################################################


# Update fig
# @callback(
#     Output(component_id='fig-pg1', component_property='figure'),
#     Input("submit", "n_clicks"),
#     State("dropdown_tickers", "value"),
# )


# def stock_price(n_clicks, ticker): 
    
#     if not n_clicks or not ticker:
#         raise PreventUpdate
    
#     df = yf.download(ticker, start='2020-01-01', end=dt.today().strftime("%Y-%m-%d"))
    
#     df.index = df.index.tz_localize(None)
    
#     df['Close'] = df['Close'].round(2)
    
#     return plot_data(df)
    

# def plot_data(df):
    
#     fig = go.Figure(layout=my_figlayout)
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'],  mode = "lines+markers",line=my_linelayout))
#     fig.update_layout(title=f'Data Linechart', xaxis_title='Time', yaxis_title='Close', height = 500)
#     fig.update_xaxes(
#         tickformat="%Y-%m-%d" 
#     )
#     fig.update_yaxes(
#         tickformat=".2f"
#     )
    
    

#     return fig

  # at the top of your file

import plotly.express as px

@callback(
    Output("fig-pg1", "figure"),
    Input("dropdown_tickers", "value"),
    Input("date-slider", "value"),
)
def stock_price(ticker, months_back):
    if not ticker or not months_back or len(ticker.strip()) < 2:
        raise PreventUpdate
    
    # Calculate start date based on months_back
    end_date = dt.today()
    start_date = end_date - pd.DateOffset(months=months_back)

    # 1) Download
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data found.")
        return fig
    
    df.index = df.index.tz_localize(None)
    df["Close"] = df["Close"].round(2)

    # 2) Flatten the columns
    df.columns = df.columns.get_level_values(0)

    # 3) Build PX line
    fig = px.line(
        df,
        x=df.index,
        y="Close",
        markers=True,
        title=f"{ticker.upper()} Close Price - Last {months_back} Month(s)",
    )

    # 4) Apply your custom figure layout
    fig.update_layout(**my_figlayout.to_plotly_json())

    # 5) Apply your custom trace style
    fig.update_traces(line=my_linelayout)

    # 6) Tidy up axes
    fig.update_xaxes(tickformat="%b %d, %Y", autorange=True)
    fig.update_yaxes(tickformat=".2f", autorange=True)

    return fig





