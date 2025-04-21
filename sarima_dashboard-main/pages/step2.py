import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from urllib.parse import parse_qs

from assets.fig_layout import my_figlayout, my_linelayout


# ──────────────────────────────────────────────────────────
# Page registration
# ──────────────────────────────────────────────────────────
dash.register_page(
    __name__,
    path="/2-ML",                       #  ← Step‑1 links here
    name="2-ML",
    title="Stock Prediction | RNN"
)


# ──────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Location(id="url", refresh=False),

    # Title
    dbc.Row(dbc.Col(html.H3("2 – RNN Stock Forecast"),
                    width=12,
                    className="row-titles")),

    # Graph
    dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id="fig-rnn",
                                          config={"displayModeBar": False})),
                    width=12),
            className="row-content")
])


# ──────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────
def _ticker_from_url(search: str) -> str | None:
    qs = parse_qs(search.lstrip("?")) if search else {}
    return qs.get("ticker", [None])[0]


@callback(
    Output("fig-rnn", "figure"),
    Input("url", "search"),
)
def rnn_forecast(search):
    # 1) ticker from query string
    ticker = _ticker_from_url(search)
    if not ticker:
        raise PreventUpdate

    # 2) download daily closes
    df = yf.download(ticker,
                     start="2020-01-01",
                     end=dt.today().strftime("%Y-%m-%d"),
                     progress=False)
    df.index = df.index.tz_localize(None)
    df = df[["Close"]].round(2)

    # 3) scale & window
    vals = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler().fit(vals)
    scaled = scaler.transform(vals)

    lookback = 20
    X, y_true = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y_true.append(scaled[i, 0])
    X = np.array(X)[:, :, None]          # (N, lookback, 1)
    y_true = np.array(y_true)            # (N,)

    # 4) train simple LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(lookback, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile("adam", "mse")
    model.fit(X, y_true, epochs=5, batch_size=16, verbose=0)

    # 5) predict one‑step sequence
    y_pred_scaled = model.predict(X).flatten()
    y_pred = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    dates = df.index[lookback:]

    # 6) MAE on overlapping part
    actual = df["Close"].values[lookback:]    # (N,) after flatten
    mae = np.mean(np.abs(actual - y_pred))

    # 7) build tidy DF for Plotly Express  ← **FIX here**
    actual_flat = df["Close"].values.flatten()
    pred_flat   = y_pred.flatten()

    plot_df = pd.DataFrame({
        "Date":   np.concatenate([df.index.values, dates]),
        "Value":  np.concatenate([actual_flat,      pred_flat]),
        "Series": ["Actual"]*len(actual_flat) + ["Predicted"]*len(pred_flat),
        "MAE":    [None]*len(actual_flat)    + [mae]*len(pred_flat)
    })

    # 8) PX line
    fig = px.line(
        plot_df,
        x="Date", y="Value",
        color="Series",
        markers=True,
        title=f"{ticker.upper()} – Actual vs. RNN Forecast",
        custom_data=["MAE"]
    )

    # 9) custom styling
    fig.layout = my_figlayout
    fig.data[0].line = my_linelayout
    fig.data[1].line = {**my_linelayout, "dash": "dash"}
    fig.data[1].hovertemplate = (
        "Date: %{x|%b %d, %Y}<br>"
        "Predicted: %{y:.2f}<br>"
        "MAE: %{customdata[0]:.2f}<extra></extra>"
    )
    fig.update_xaxes(tickformat="%b %d, %Y")
    fig.update_yaxes(tickformat=".2f")

    return fig


