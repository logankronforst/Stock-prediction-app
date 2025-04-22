# pages/step3.py

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

from assets.fig_layout import my_figlayout, my_linelayout

dash.register_page(
    __name__,
    path="/3-gradient-descent",
    name="3 – Gradient Descent",
    title="Stock Prediction | GD Visualization"
)

def cost_fn(x):
    return x**2

def compute_adam_path(initial=10.0, lr=0.1, steps=100, tol=1e-3):
    x = tf.Variable(initial, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    hist = []
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = cost_fn(x)
        grad = tape.gradient(loss, x)
        opt.apply_gradients([(grad, x)])
        xi, yi = float(x.numpy()), float(loss.numpy())
        hist.append((xi, yi))
        if abs(xi) < tol:
            break
    return np.array(hist)

layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H3("3 – Gradient Descent via Adam"), className="row-titles")),
    dbc.Row(dbc.Col(dcc.Graph(id="fig-gd3"), width=12), className="row-content")
])

@callback(
    Output("fig-gd3", "figure"),
    Input("browser-memo", "data")
)
def draw_gradient_descent(store):
    lr    = float(store.get("gd_lr", 0.1))
    steps = int(store.get("gd_steps", 100))

    traj = compute_adam_path(initial=10.0, lr=lr, steps=steps, tol=1e-3)
    xs, ys = traj[:,0], traj[:,1]

    # prepare parabola
    x_line = np.linspace(-10, 10, 400)
    y_line = x_line**2

    fig = go.Figure(layout=my_figlayout)

    # trace 0: neon‑green parabola
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=my_linelayout,
        name="f(x)=x²"
    ))

    # trace 1: initial red dot
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Adam step"
    ))

    # build frames—only update trace 1, plus a styled loss annotation
    frames = []
    for i, (xv, yv) in enumerate(zip(xs, ys)):
        ann = dict(
            xref="paper", yref="paper",
            x=0.15, y=0.98,                         # pushed right
            text=f"<b>Loss:</b> {yv:.4f}",
            showarrow=False,
            font=dict(color=my_linelayout["color"], size=16, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.6)",              # semi‑opaque dark
            bordercolor=my_linelayout["color"],     # neon border
            borderwidth=2,
            borderpad=6
        )
        frames.append(go.Frame(
            name=str(i),
            data=[go.Scatter(x=[xv], y=[yv], mode="markers",
                             marker=dict(color="red", size=10))],
            traces=[1],  # only update the red‐dot
            layout=go.Layout(annotations=[ann])
        ))
    fig.frames = frames

    # initial annotation
    fig.update_layout(annotations=[dict(
        xref="paper", yref="paper",
        x=0.15, y=0.98,
        text=f"<b>Loss:</b> {ys[0]:.4f}",
        showarrow=False,
        font=dict(color=my_linelayout["color"], size=16, family="Arial Black"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=my_linelayout["color"],
        borderwidth=2,
        borderpad=6
    )])

    # play‐only controls
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.05, x=1.15, xanchor="right", yanchor="top",
            buttons=[dict(
                label=" Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
            )]
        )],
        xaxis_title="x",
        yaxis_title="f(x)",
        height=600
    )

    # lock axes to parabola range
    fig.update_xaxes(range=[x_line.min(), x_line.max()], autorange=False)
    fig.update_yaxes(range=[0, y_line.max()], autorange=False)

    return fig
