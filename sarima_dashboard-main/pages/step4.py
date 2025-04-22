# pages/step4.py

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

from assets.fig_layout import my_figlayout, my_linelayout

dash.register_page(
    __name__,
    path="/4-hyperparam-tuner",
    name="4 – Hyperparameter Tuner",
    title="Stock Prediction | Hyperparameter Tuner"
)

layout = dbc.Container(fluid=True, children=[

    # Title
    dbc.Row(dbc.Col(html.H3("4 – Hyperparameter Tuner"),
                    className="row-titles")),

    # Controls + Train button
    dbc.Row([

        # ── Left column: sliders & button
        dbc.Col([
            html.Label("Number of LSTM Layers:", style={"color": "#3DED97"}),
            dcc.Slider(1, 3,
                       step=1,
                       marks={1:"1", 2:"2", 3:"3"},
                       value=1,
                       id="num-layers"),
            html.Br(),

            html.Label("Units in Layer 1:", style={"color": "#3DED97"}),
            dcc.Slider(5, 128,
                       step=None,
                       marks={5:"5", 32:"32", 64:"64", 128:"128"},
                       value=32,
                       id="units-1"),
            html.Br(),

            html.Label("Units in Layer 2:", style={"color": "#3DED97"}),
            dcc.Slider(5, 128,
                       step=None,
                       marks={5:"5", 32:"32", 64:"64", 128:"128"},
                       value=32,
                       id="units-2"),
            html.Br(),

            html.Label("Units in Layer 3:", style={"color": "#3DED97"}),
            dcc.Slider(5, 128,
                       step=None,
                       marks={5:"5", 32:"32", 64:"64", 128:"128"},
                       value=32,
                       id="units-3"),
            html.Hr(style={"borderColor":"rgba(255,255,255,0.2)"}),

            html.Label("Learning Rate:", style={"color": "#3DED97"}),
            dcc.Slider(1e-4, 1e-1,
                       step=None,
                       marks={1e-4:"1e-4", 1e-2:"1e-2", 1e-1:"1e-1"},
                       value=1e-2,
                       id="learning-rate"),
            html.Br(),

            html.Label("Epochs:", style={"color": "#3DED97"}),
            dcc.Slider(1, 20,
                       step=1,
                       marks={1:"1", 5:"5", 10:"10", 20:"20"},
                       value=5,
                       id="epochs"),
            html.Br(),

            dbc.Button("Train", id="train-btn",
                       color="success", className="mt-2"),
        ], width=4, className="div-hyperpar"),


        # ── Right column: architecture & forecast plots
        dbc.Col([
            dbc.Row(dcc.Loading(
                dcc.Graph(id="fig-arch", config={"displayModeBar": False})
            ), className="mb-4"),
            dbc.Row(dcc.Loading(
                dcc.Graph(id="fig-forecast", config={"displayModeBar": False})
            )),
        ], width=8),

    ], className="row-content"),

    # this Store is written in step1, read here to pull the ticker
    dcc.Store(id="fig-step1-store", storage_type="session")
])


@callback(
    Output("fig-forecast", "figure"),
    Input("fig-step1-store", "data"),    # <-- fixed here
    Input("train-btn", "n_clicks"),
    State("num-layers", "value"),
    State("units-1", "value"),
    State("units-2", "value"),
    State("units-3", "value"),
    State("learning-rate", "value"),
    State("epochs", "value"),
)
def update_rnn(store, n_clicks, nl, u1, u2, u3, lr, epochs):
    # ── 1) determine ticker (TSLA if none in store)
    ticker = (store or {}).get("ticker", "TSLA")

    # ── 2) pull & prep exactly like step2
    df = yf.download(
        ticker,
        start="2020-01-01",
        end=dt.today().strftime("%Y-%m-%d"),
        progress=False
    )
    df.index = df.index.tz_localize(None)
    closes = df["Close"].round(2).values.reshape(-1,1)

    scaler = MinMaxScaler().fit(closes)
    scaled = scaler.transform(closes)
    lookback = 20
    X, y_true = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i,0])
        y_true.append(scaled[i,0])
    X = np.array(X)[:,:,None]
    y_true = np.array(y_true)

    # ── 3) build the variable‐size LSTM
    units = [u1, u2, u3][:nl]
    model = tf.keras.Sequential()
    for i,u in enumerate(units):
        model.add(tf.keras.layers.LSTM(
            u,
            return_sequences=(i<nl-1),
            input_shape=(lookback,1) if i==0 else None
        ))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    model.fit(X, y_true,
              epochs=epochs,
              batch_size=16,
              verbose=0)

    # ── 4) forecast
    y_pred = scaler.inverse_transform(
        model.predict(X).reshape(-1,1)
    ).flatten()
    dates = df.index[lookback:]

    # ── 5) build your forecast figure
    plot_df = pd.DataFrame({
        "Date":   np.concatenate([df.index.values, dates]),
        "Value":  np.concatenate([closes.flatten(), y_pred]),
        "Series": ["Actual"]*len(closes) + ["Predicted"]*len(y_pred)
    })
    fig_f = px.line(
        plot_df,
        x="Date", y="Value", color="Series",
        markers=True,
        color_discrete_map={
            "Actual":    my_linelayout["color"],
            "Predicted": "#ff7b00"
        }
    )
    fig_f.layout = my_figlayout
    fig_f.data[0].line = my_linelayout
    fig_f.data[1].line.width = my_linelayout["width"]
    fig_f.data[1].line.dash  = "dash"
    fig_f.update_xaxes(tickformat="%b %d, %Y")
    fig_f.update_yaxes(tickformat=".2f")

    return fig_f


@callback(
    Output("fig-arch", "figure"),
    Input("num-layers", "value"),
    Input("units-1", "value"),
    Input("units-2", "value"),
    Input("units-3", "value"),
)
def update_architecture(nl, u1, u2, u3):
    lookback = 20
    units = [u1, u2, u3][:nl]
    layers = [lookback] + units + [1]
    xs = np.linspace(0, 1, len(layers))
    Xs, Ys, links = [], [], []

    for i, n in enumerate(layers):
        y_coords = np.linspace(0, 1, n + 2)[1:-1]
        for y in y_coords:
            Xs.append(xs[i])
            Ys.append(y)
        if i < len(layers) - 1:
            next_y = np.linspace(0, 1, layers[i+1] + 2)[1:-1]
            for y0 in y_coords:
                for y1 in next_y:
                    links.append(((xs[i], y0), (xs[i+1], y1)))

    fig_a = go.Figure(layout=my_figlayout)
    for (x0, y0), (x1, y1) in links:
        fig_a.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="rgba(62,180,137,0.3)"),
            showlegend=False
        ))
    fig_a.add_trace(go.Scatter(
        x=Xs, y=Ys,
        mode="markers",
        marker=dict(
            size=20,
            color=my_linelayout["color"],
            line=dict(color="white", width=1)
        ),
        showlegend=False
    ))
    fig_a.update_xaxes(visible=False)
    fig_a.update_yaxes(visible=False)
    fig_a.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig_a


