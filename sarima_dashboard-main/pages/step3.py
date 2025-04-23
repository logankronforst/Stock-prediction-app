# pages/step3.py

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

from assets.fig_layout import my_figlayout
try:
    from pages.step2 import train_rnn_model
except ImportError:
    # Fallback stub: returns a constant zero-loss history
    def train_rnn_model(lr, steps):
        return [0.0 for _ in range(steps)]

dash.register_page(
    __name__,
    path="/3-gradient-descent",
    name="3 – Gradient Descent via Adam",
    title="Stock Prediction | GD Visualization"
)


def cost_fn(x, y):
    # simple convex paraboloid surface
    return x**2 + 0.5 * y**2


def compute_adam_path(initial=(3.0, 3.0), lr=0.1, steps=50, tol=1e-3):
    x = tf.Variable(initial, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    hist = []
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = cost_fn(x[0], x[1])
        grad = tape.gradient(loss, x)
        opt.apply_gradients([(grad, x)])
        xi, yi, zi = float(x[0].numpy()), float(x[1].numpy()), float(loss.numpy())
        hist.append((xi, yi, zi))
        if tf.norm(x) < tol:
            break
    return np.array(hist)


layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H3("3 – Gradient Descent via Adam"), className="row-titles")),
    dbc.Row(dbc.Col(
        dcc.Loading(
            dcc.Graph(
                id="fig-gd3",
                style={
                    "width": "100vw",      # full width
                    "height": "100vh",     # taller
                    "margin": "0 auto",
                    "display": "block"
                }
            ),
            type="default",
            color="rgb(61,237,151)"
        ),
        width=12
    ), className="row-content")
])


@callback(
    Output("fig-gd3", "figure"),
    Input("browser-memo", "data")
)
def draw_gradient_descent(store):
    lr = float(store.get("gd_lr", 0.1))
    steps = int(store.get("gd_steps", 50))

    # get RNN training loss history using same lr and step count
    # model_losses = train_rnn_model(lr=lr, steps=steps)

    traj = compute_adam_path(initial=(3.0, 3.0), lr=lr, steps=steps, tol=1e-3)
    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

    # Use parabola loss history if model_losses is stubbed
    model_losses = zs  # override with actual cost_fn losses so annotations update

    # per-feature loss contributions
    loss_xs = xs**2
    loss_ys = 0.5 * ys**2

    # surface grid
    grid = np.linspace(-3, 3, 200)
    Xg, Yg = np.meshgrid(grid, grid)
    Zg = cost_fn(
        tf.constant(Xg, tf.float32),
        tf.constant(Yg, tf.float32)
    ).numpy()

    # initialize figure with transparent background and neon grid
    fig = go.Figure(layout=my_figlayout)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(61,237,151,0.2)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(61,237,151,0.2)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(61,237,151,0.2)'),
            camera=dict(eye=dict(x=0, y=0, z=4))  # high overhead
        )
    )

    # multicolored Jet surface
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=Zg,
        showscale=False,
        colorscale='Jet',
        opacity=0.8
    ))

    # initial path trace (will be animated)
    fig.add_trace(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]],
        mode="lines+markers",
        line=dict(color="rgb(61,237,151)", width=4),
        marker=dict(size=6, color="rgb(61,237,151)"),
        name="Loss Path",
        showlegend=True
    ))

    # initial loss annotation
    initial_ann = dict(
        xref="paper", yref="paper",
        x=0.15, y=0.88,
        text=(
            f"<b>Training Loss:</b> {model_losses[0]:.4f}"
        ),
        showarrow=False,
        font=dict(color="rgb(61,237,151)", size=16),
        bgcolor="rgba(0,0,0,1)",
        bordercolor="rgb(61,237,151)", borderwidth=2, borderpad=10,
    )
    fig.update_layout(annotations=[initial_ann])

    # build animation frames for path and loss
    frames = []
    for i in range(1, len(xs)):
        path_frame = go.Scatter3d(
            x=xs[:i+1], y=ys[:i+1], z=zs[:i+1],
            mode="lines+markers",
            line=dict(color="rgb(61,237,151)", width=4),
            marker=dict(size=6, color="rgb(61,237,151)")
        )
        ann = dict(initial_ann)
        ann["text"] = (
            f"<b>Training Loss:</b> {model_losses[i]:.4f}"
        )
        frames.append(go.Frame(
            name=str(i),
            data=[path_frame],
            traces=[1],
            layout=go.Layout(annotations=[ann])
        ))
    fig.frames = frames

    # play button, centered, legend below, wider plot
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.1, x=0.5,
            xanchor="center", yanchor="top",
            buttons=[dict(
                label=" Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
            )]
        )],
        legend=dict(
            title="",
            x=0.5, y=0.92,
            xanchor="center", yanchor="top",
            orientation="h",
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,  # allow full area usage
        
        height=1000
    )

    return fig