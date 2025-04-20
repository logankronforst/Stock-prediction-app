import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='Stock Prediction | Home')

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3(['Welcome!']),
            html.P([html.B(['App Overview'])], className='par')
        ], width=12, className='row-titles')
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            html.P([html.B('1) Stock Indicators'),html.Br(),
                    'Input a stock code and visualize its EWA and adjusted close across time.'], className='guide'),
            html.P([html.B('2) Predict a stock adjusted close value with machine learning.'),html.Br(),
                    'Select from two different neural networks to predict any stocks value.'], className='guide'),
            html.P([html.B('3) Visualize gradient descent in real time.'),html.Br(),
                    'Visualize the gradient descent and optimization proceedures in three dimenions.',html.Br(),
                    ''], className='guide'),
            html.P([html.B('4) Tune your neural network'),html.Br(),
                    'Adjust the hyperparameters of your neural network',html.Br(),
                    '',html.Br(),
                    ''], className='guide')
        ], width = 8),
        dbc.Col([], width = 2)
    ])
])