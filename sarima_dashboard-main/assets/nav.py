# assets/nav.py
import dash
from dash import html
import dash_bootstrap_components as dbc

_nav = dbc.Container(fluid=True, children=[
    # logo + title row
    dbc.Row([
        dbc.Col(
            html.Div(html.I(className="fa-solid fa-chart-simple fa-2x"),
                     className="logo"),
            width=2,
        ),
        dbc.Col(html.H1("Stock Prediction", className="app-brand"),
                width=10),
    ], className="mb-4"),
    # nav‐links row
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavLink(
                        page["name"],
                        href=page["path"],
                        active="exact",
                        className="my-nav-link"
                    )
                    for page in dash.page_registry.values()
                    # you can filter out 404 etc here if you like:
                    if page["module"] != "pages.notfound"
                ],
                vertical=True,
                pills=True,
                className="my-nav"
            ),
            width=12,
        )
    ])
])

