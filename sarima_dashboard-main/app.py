import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# register pages automatically from pages/*.py
# make sure that your pages folder is laid out like:
#   pages/
#     __init__.py
#     step1.py
#     step2.py
#     step3.py
#     step4.py
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,  # we have callbacks in pages
)
server = app.server

# your shared nav and footer components
from assets.nav import _nav
from assets.footer import _footer

app.layout = dbc.Container(fluid=True, children=[
    # capture the URL so callbacks in pages can read it
    dcc.Location(id="url", refresh=False),

    dbc.Row([
        # sidebar
        dbc.Col(_nav, width=2, style={"padding": 0}),
        # page content
        dbc.Col(dash.page_container, width=10)
    ], className="h-100"),

    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col(_footer, width=10)
    ]),

    # session‑wide store for the ticker & anything else
    dcc.Store(id="browser-memo", data={}, storage_type="session")

], style={"height": "100vh", "padding": 0})


if __name__ == "__main__":
    app.run(debug=True)
