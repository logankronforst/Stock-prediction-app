# pages/step1.py

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from datetime import datetime as dt

from assets.fig_layout import my_figlayout, my_linelayout

dash.register_page(
    __name__,
    path="/1-input",
    name="1 – Input Stock Code",
    title="Stock Prediction | Step 1"
)

layout = dbc.Container(fluid=True, children=[

    # Title
    dbc.Row(dbc.Col(html.H3("1 – Input Stock Code"), className="row-titles")),

    # Ticker + Slider
    dbc.Row(
        [
            dbc.Col([], width=2),
            dbc.Col(html.P("Stock Ticker:"), width=2),
            dbc.Col(
                dbc.Input(
                    id="dropdown_tickers",
                    type="text",
                    placeholder="e.g. AAPL",
                    value="",         # start empty → fallback to TSLA
                    debounce=True,
                    style={"color":"#000","backgroundColor":"#fff"}
                ), width=3
            ),
            dbc.Col(html.P("History:"), width=1, style={"paddingTop":"5px"}),
            dbc.Col(
                dcc.Slider(
                    id="date-slider",
                    min=1, max=24, step=None,
                    marks={1:"1M",3:"3M",6:"6M",12:"1Y",24:"2Y"},
                    value=6,           # default 6 months
                    tooltip={"always_visible":False}
                ), width=4
            ),
            dbc.Col([], width=2),
        ],
        className="row-content",
        justify="center"
    ),

    # hidden stores
    dcc.Store(id="fig-step1-store", storage_type="session"),
    # global store in app.layout:
    # dcc.Store(id="browser-memo", data={}, storage_type='session')

    # Graph
    dbc.Row(
        dbc.Col(
            dcc.Loading(dcc.Graph(id="fig-pg1", config={"displayModeBar":False})),
            width=12
        ),
        className="row-content"
    ),

])


# 1) Whenever ticker or slider changes, recompute & cache the figure + user inputs
@callback(
    Output("fig-step1-store","data"),
    Output("browser-memo","data"),
    Input("dropdown_tickers","value"),
    Input("date-slider","value"),
)
def cache_step1(ticker: str, months_back: int):
    # require a nonempty ticker to cache; otherwise leave store untouched
    if not ticker or len(ticker.strip())<1:
        return no_update, no_update

    t = ticker.strip().upper()
    # compute date window
    end = dt.today()
    start = end - pd.DateOffset(months=months_back)

    # download
    df = yf.download(t, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        # if no data found, clear the step1 store so fallback will trigger
        return None, None

    df.index = df.index.tz_localize(None)
    df["Close"] = df["Close"].round(2)
    # flatten multi‐level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # build PX line
    fig = px.line(
        df,
        x=df.index, y="Close",
        markers=True,
        title=f"{t} Close — Last {months_back} Month(s)"
    )
    fig.layout = my_figlayout
    fig.data[0].line = my_linelayout
    fig.update_xaxes(tickformat="%b %d, %Y", autorange=True)
    fig.update_yaxes(tickformat=".2f", autorange=True)

    # cache and also write to global store
    return fig.to_dict(), {"ticker": t, "months": months_back}


# 2) Display whatever’s in our cache—or fallback to TSLA/6M on first load
@callback(
    Output("fig-pg1","figure"),
    Input("fig-step1-store","data"),
)
def display_step1(fig_json):
    if fig_json:
        return fig_json

    # fallback → TSLA, last 6 months
    end = dt.today()
    start = end - pd.DateOffset(months=6)
    df = yf.download("TSLA", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    df.index = df.index.tz_localize(None)
    df["Close"] = df["Close"].round(2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    fig = px.line(
        df,
        x=df.index, y="Close",
        markers=True,
        title="TSLA (default) Close — Last 6 Month(s)"
    )
    fig.layout = my_figlayout
    fig.data[0].line = my_linelayout
    fig.update_xaxes(tickformat="%b %d, %Y", autorange=True)
    fig.update_yaxes(tickformat=".2f", autorange=True)
    return fig.to_dict()
