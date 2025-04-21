import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from datetime import datetime as dt

from assets.fig_layout import my_figlayout, my_linelayout

# Give this page its own path so it doesn't conflict with 'home'
dash.register_page(
    __name__,
    path="/step1",
    name="1-Input Stock Code",
    title="Stock Prediction | Step 1"
)

layout = dbc.Container(fluid=True, children=[

    # Title row
    dbc.Row(
        dbc.Col(html.H3("1 – Input Stock Code"), width=12, className="row-titles")
    ),

    # Input row
    dbc.Row(
        [
            dbc.Col([], width=2),
            dbc.Col(html.P("Input stock code:"), width=2),
            dbc.Col(
                [
                    dbc.Input(
                        id="dropdown_tickers",
                        type="text",
                        placeholder="e.g. AAPL",
                        style={"color": "#000", "backgroundColor": "#fff"}  # force dark text
                    ),
                    dbc.Button(
                        "Submit",
                        id="submit",
                        outline=True,
                        color="Success",
                        className="mt-2"
                    ),
                ],
                width=4,
            ),
            dbc.Col([], width=4),
        ],
        className="input-place mb-4",
    ),

    # Store to cache the figure in sessionStorage
    dcc.Store(id="fig-step1-store", storage_type="session"),

    # Graph row (no persistence prop here)
    dbc.Row(
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id="fig-pg1",
                    config={"displayModeBar": False}
                )
            ),
            width=12
        ),
        className="row-content"
    ),

    # Link to Step 2
    dbc.Row(
        [
            dbc.Col([], width=2),
            dbc.Col(
                dcc.Link(
                    "→ Next: Machine Learning",
                    href="",  # populated by callback
                    id="to-step2",
                    className="btn btn-outline-primary mt-3",
                    style={"textDecoration": "none"}
                ),
                width=4
            ),
            dbc.Col([], width=6),
        ],
        className="row-content"
    ),

])


# 1) Compute & cache the figure when the user clicks Submit
@callback(
    Output("fig-step1-store", "data"),
    Input("submit", "n_clicks"),
    State("dropdown_tickers", "value"),
)
def compute_step1_figure(n_clicks, ticker):
    if not n_clicks or not ticker:
        raise PreventUpdate

    # Download and clean the data
    df = yf.download(
        ticker,
        start="2020-01-01",
        end=dt.today().strftime("%Y-%m-%d"),
        progress=False
    )
    df.index = df.index.tz_localize(None)
    df["Close"] = df["Close"].round(2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build the PX line chart
    fig = px.line(
        df,
        x=df.index,
        y="Close",
        markers=True,
        title=f"{ticker.upper()} Close Price"
    )

    # Reapply custom layouts
    fig.layout = my_figlayout
    fig.data[0].line = my_linelayout
    fig.update_xaxes(tickformat="%b %d, %Y", autorange=True)
    fig.update_yaxes(tickformat=".2f", autorange=True)

    # Store the figure JSON
    return fig.to_dict()


# 2) Read and display the cached figure
@callback(
    Output("fig-pg1", "figure"),
    Input("fig-step1-store", "data"),
)
def display_step1_figure(fig_dict):
    if not fig_dict:
        raise PreventUpdate
    return fig_dict


# 3) Populate the “Next” link with the ticker as a query param
@callback(
    Output("to-step2", "href"),
    Input("submit", "n_clicks"),
    State("dropdown_tickers", "value"),
)
def update_step2_link(n_clicks, ticker):
    if n_clicks and ticker:
        return f"/2-ML?ticker={ticker}"
    return "/2-ML"


