# pages/step2.py

import dash
from dash import dcc, callback, Input, Output, html
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from assets.fig_layout import my_figlayout, my_linelayout

dash.register_page(
    __name__,
    path="/2-ML",
    name="2 – Machine Learning",
    title="Stock Prediction | RNN Forecast"
)

layout = dbc.Container(fluid=True, children=[
    # Title
    dbc.Row(dbc.Col(html.H3("2 – RNN Stock Forecast"), className="row-titles")),

    # New input: how many days to forecast into the future
    dbc.Row([
        dbc.Col(html.Label("Forecast Days Ahead:", style={"color": "#3DED97"}), width=2),
        dbc.Col(dcc.Input(
            id="future-days",
            type="number",
            min=1,
            value=5,
            style={"width": "100%", "color": "#3DED97"}
        ), width=2),
    ], className="row-content"),

    # RNN forecast plot
    dbc.Row(
        dbc.Col(
            dcc.Loading(
                dcc.Graph(id="fig-rnn", config={"displayModeBar": False})
            ), width=12
        ),
        className="row-content"
    ),
])


@callback(
    Output("fig-rnn", "figure"),
    Input("browser-memo", "data"),
    Input("future-days", "value")
)
def rnn_page(store, future_days):
    # 1) pull ticker & months from store (fallbacks)
    t = store.get("ticker", "TSLA")
    m = int(store.get("months", 6))

    # 2) download data
    end = dt.today()
    start = end - pd.DateOffset(months=m)
    df = yf.download(
        t,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False
    )
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to model")
        return fig

    df.index = df.index.tz_localize(None)
    closes = df["Close"].round(2).values.reshape(-1, 1)

    # 3) scale data
    scaler = MinMaxScaler().fit(closes)
    scaled = scaler.transform(closes)

    # 4) create LSTM windows
    lookback = 20
    X, y_true = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y_true.append(scaled[i, 0])
    X = np.array(X)[:, :, None]
    y_true = np.array(y_true)

    # 5) build & train the RNN
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(lookback, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y_true, epochs=5, batch_size=16, verbose=0)

    # 6) predict on training window and invert scale
    y_pred_scaled = model.predict(X).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    dates_pred = df.index[lookback:]

    # 7) compute MAE
    mae = np.mean(np.abs(closes.flatten()[lookback:] - y_pred))

    # 8) prepare DataFrame for plotting actual vs. predicted
    actual_flat = closes.flatten()
    predicted_flat = y_pred
    df_plot = pd.DataFrame({
        "Date":   np.concatenate([df.index.values, dates_pred]),
        "Value":  np.concatenate([actual_flat, predicted_flat]),
        "Series": ["Actual"] * len(actual_flat) + ["Predicted"] * len(predicted_flat),
        "MAE":    [None] * len(actual_flat) + [mae] * len(predicted_flat)
    })

    # 9) forecast future 'x' days if requested
    future_days = int(future_days or 0)
    if future_days > 0:
        last_seq = scaled[-lookback:].reshape(1, lookback, 1)
        future_scaled = []
        for _ in range(future_days):
            next_scale = model.predict(last_seq).flatten()[0]
            future_scaled.append(next_scale)
            # roll the window
            last_seq = np.concatenate(
                (last_seq[:, 1:, :], np.array(next_scale).reshape(1, 1, 1)),
                axis=1
            )
        future = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()
        future_dates = pd.date_range(
            dates_pred[-1] + pd.Timedelta(days=1),
            periods=future_days,
            freq="B"
        )
        df_future = pd.DataFrame({
            "Date":   future_dates,
            "Value":  future,
            "Series": ["Future"] * future_days,
            "MAE":    [None] * future_days
        })
        df_plot = pd.concat([df_plot, df_future], ignore_index=True)
        color_map = {
            "Actual":    my_linelayout["color"],
            "Predicted": "#ff7b00",
            "Future":    "#00ccff"
        }
    else:
        color_map = {
            "Actual":    my_linelayout["color"],
            "Predicted": "#ff7b00"
        }

    # 10) final plot
    fig = px.line(
        df_plot,
        x="Date", y="Value",
        color="Series", markers=True,
        custom_data=["MAE"],
        title=f"{t} – Actual vs RNN Forecast",
        color_discrete_map=color_map
    )
    fig.layout = my_figlayout
    # style traces
    for trace in fig.data:
        if trace.name == "Predicted":
            trace.line.width = my_linelayout["width"]
            trace.line.dash = "dash"
        if trace.name == "Future":
            trace.line.dash = "dot"
            trace.marker.size = 8

    fig.update_xaxes(tickformat="%b %d, %Y")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        legend=dict(orientation="h", y=1.02, yanchor="bottom", xanchor="right", x=1)
    )

    return fig