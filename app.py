# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:43:06 2023

@author: rhanusa
"""
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dashboard_components as dc
            
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)
app.layout = dbc.Container([
    dbc.Row([dbc.Col(dcc.Graph(id="energy_allocation_graph",figure=dc.fig_e_allo)),
    dbc.Row([dbc.Col(dcc.Graph(id="reactor1_output_graph", figure=dc.fig_r1), width=6),
            dbc.Col(dcc.Graph(id="reactor2_output_graph", figure=dc.fig_r2), width=6,)],
            style=dict(padding="0px", margin="0px")),
    dbc.Row([dbc.Col(dcc.Graph(id="r1_rxn_graph", figure=dc.fig_r1_rxn), width=6),
             dbc.Col(dcc.Graph(id="r2_rxn_graph", figure=dc.fig_r2_rxn), width=6)])
    ]),
    dcc.Interval(id="interval",interval=10, 
                  n_intervals=0, max_intervals=len(dc.r2_sx_out)-501)])


@app.callback(Output(component_id='reactor1_output_graph', component_property='extendData'),
              Output(component_id='reactor2_output_graph', component_property='extendData'),
              Output(component_id='r1_rxn_graph', component_property='extendData'),
              Output(component_id='r2_rxn_graph', component_property='extendData'),
              Output(component_id='energy_allocation_graph', component_property='extendData'),
              Input(component_id='interval', component_property='n_intervals'))

def update(n_intervals):
    return dc.update(n_intervals)

if __name__ == "__main__":
    app.run_server(debug=True)