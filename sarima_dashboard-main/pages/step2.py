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

    # split in half for train/test
    half = len(closes) // 2

    # 3) scale data on training portion
    scaler = MinMaxScaler().fit(closes[:half])
    scaled_full = scaler.transform(closes)

    lookback = 20
    # training windows
    X_train, y_train = [], []
    for i in range(lookback, half):
        X_train.append(scaled_full[i-lookback:i, 0])
        y_train.append(scaled_full[i, 0])
    X_train = np.array(X_train)[:, :, None]
    y_train = np.array(y_train)

    # testing windows
    X_test, y_test = [], []
    for i in range(half, len(scaled_full)):
        X_test.append(scaled_full[i-lookback:i, 0])
        y_test.append(scaled_full[i, 0])
    X_test = np.array(X_test)[:, :, None]
    y_test = np.array(y_test)

    # 5) build & train the RNN
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(lookback, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    # predict on test set and inverse scale
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    # true test values back to original scale
    y_true_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    # compute per-sample absolute error
    losses = np.abs(y_true_test - y_pred)
    # corresponding dates for test
    dates_test = df.index[half:]

    # 9) forecast future 'x' days if requested
    future_days = int(future_days or 0)
    if future_days > 0:
        last_seq = scaled_full[-lookback:].reshape(1, lookback, 1)
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
            dates_test[-1] + pd.Timedelta(days=1),
            periods=future_days,
            freq="B"
        )
        df_future = pd.DataFrame({
            "Date":   future_dates,
            "Value":  future,
            "Series": ["Future"] * future_days,
            "Loss":    [None] * future_days
        })
        color_map = {
            "Actual":    my_linelayout["color"],
            "Predicted": "#ff7b00",
            "Future":    "#00ccff",
            "Actual Full": "#888888",
        }
    else:
        color_map = {
            "Actual":    my_linelayout["color"],
            "Predicted": "#ff7b00",
            "Actual Full": "#888888",
        }

    # actual-only for first half
    df_actual = pd.DataFrame({
        "Date": df.index[:half],
        "Value": closes.flatten()[:half],
        "Series": ["Actual"] * half,
        "Loss": [None] * half
    })
    # full-period actual series
    df_full = pd.DataFrame({
        "Date": df.index,
        "Value": closes.flatten(),
        "Series": ["Actual Full"] * len(df),
        "Loss": [None] * len(df)
    })
    # predicted vs actual on second half
    df_pred = pd.DataFrame({
        "Date": dates_test,
        "Value": y_pred,
        "Series": ["Predicted"] * len(y_pred),
        "Loss": losses
    })
    # combine
    df_plot = pd.concat([df_full, df_actual, df_pred], ignore_index=True)
    # append future if any
    if future_days > 0:
        df_plot = pd.concat([df_plot, df_future], ignore_index=True)

    # 10) final plot
    fig = px.line(
        df_plot,
        x="Date", y="Value",
        color="Series", markers=True,
        custom_data=["Loss"],
        title=f"{t} – Actual vs RNN Forecast",
        color_discrete_map=color_map
    )
    fig.layout = my_figlayout
    # apply custom line styles by series name
    for trace in fig.data:
        if trace.name in ["Actual", "Actual Full"]:
            trace.line.color = my_linelayout["color"]
            trace.line.width = my_linelayout["width"]
            trace.line.dash = "solid"
        elif trace.name == "Predicted":
            trace.line.color = color_map["Predicted"]
            trace.line.width = my_linelayout["width"]
            trace.line.dash = "dash"
        elif trace.name == "Future":
            trace.line.color = color_map.get("Future")
            trace.line.width = my_linelayout["width"]
            trace.line.dash = "dot"
            trace.marker.size = 8

    # show per-point loss in tooltip
    fig.update_traces(
        hovertemplate="%{x|%b %d, %Y}<br>%{series}: %{y:.2f}<br>Loss: %{customdata[0]:.4f}"  # adjust format as needed
    )

    fig.update_xaxes(tickformat="%b %d, %Y")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        legend=dict(orientation="h", y=1.02, yanchor="bottom", xanchor="right", x=1)
    )

    return fig