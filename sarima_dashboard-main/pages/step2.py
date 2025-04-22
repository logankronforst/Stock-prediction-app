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
    dbc.Row(dbc.Col(html.H3("2 – RNN Stock Forecast"), className="row-titles")),
    dbc.Row(
        dbc.Col(dcc.Loading(dcc.Graph(id="fig-rnn", config={"displayModeBar":False})), width=12),
        className="row-content"
    ),
])


@callback(
    Output("fig-rnn","figure"),
    Input("browser-memo","data")
)
def rnn_page(store):
    # fallback to TSLA/6
    t = store.get("ticker","TSLA")
    m = int(store.get("months",6))

    end   = dt.today()
    start = end - pd.DateOffset(months=m)
    df = yf.download(t, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        fig = go.Figure(); fig.update_layout(title="No data to model")
        return fig

    df.index = df.index.tz_localize(None)
    closes  = df["Close"].round(2)

    # LSTM windows
    scaled = MinMaxScaler().fit_transform(closes.values.reshape(-1,1))
    lookback = 20
    X, y_true = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y_true.append(scaled[i,0])

    X = np.array(X)[:,:,None]   # <-- now X will always be 2D→3D
    y_true = np.array(y_true)

    # quick LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(lookback,1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile("adam","mse")
    model.fit(X, y_true, epochs=5, batch_size=16, verbose=0)

    # predict + invert
    y_pred_scaled = model.predict(X).flatten()
    y_pred = MinMaxScaler().fit(closes.values.reshape(-1,1)) \
                   .inverse_transform(y_pred_scaled.reshape(-1,1)) \
                   .flatten()
    dates_pred = closes.index[lookback:]

    # MAE
    mae = np.mean(np.abs(closes.values[lookback:] - y_pred))

    # flatten everything for concat
    actual_flat    = closes.values.ravel()
    predicted_flat = y_pred.ravel()

    df_plot = pd.DataFrame({
        "Date":  np.concatenate([closes.index.values, dates_pred]),
        "Value": np.concatenate([actual_flat, predicted_flat]),
        "Series": ["Actual"]*len(actual_flat) + ["Predicted"]*len(predicted_flat),
        "MAE":   [None]*len(actual_flat)   + [mae]*len(predicted_flat)
    })

    fig = px.line(
        df_plot, x="Date", y="Value",
        color="Series", markers=True, custom_data=["MAE"],
        title=f"{t} – Actual vs RNN Forecast",
        color_discrete_map={"Actual":my_linelayout["color"], "Predicted":"#ff7b00"}
    )
    fig.layout = my_figlayout
    fig.data[0].line = my_linelayout
    fig.data[1].line.width = my_linelayout["width"]
    fig.data[1].line.dash  = "dash"
    fig.data[1].marker.size = 6
    fig.data[1].hovertemplate = (
        "Date: %{x|%b %d, %Y}<br>"
        "Predicted: %{y:.2f}<br>"
        "MAE: %{customdata[0]:.2f}<extra></extra>"
    )
    fig.update_xaxes(tickformat="%b %d, %Y")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(legend=dict(
        orientation="h", y=1.02, yanchor="bottom",
        xanchor="right", x=1
    ))

    return fig
