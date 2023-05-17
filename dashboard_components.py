# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:30:04 2023

@author: rhanusa
"""
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from weather_energy_components import data_length
from plant_components import Battery, Reactor1, Reactor2
from weather_energy_components import Hourly_state, distribute_energy

alloc_step = 10
idle_start_length = 50
x = [i for i in range(-50,1)]
r1_cos_out = np.zeros(data_length + idle_start_length)
r2_sx_out = np.zeros(data_length + idle_start_length)
r1_e = np.zeros(data_length + idle_start_length)
r2_e = np.zeros(data_length + idle_start_length)
generated_kw = np.zeros(data_length + idle_start_length)
consumed_kw = np.zeros(data_length + idle_start_length)
battery_charge = [50]*(data_length + idle_start_length)

r1_prev = 0 
r2_prev = 0 
reactor2 = Reactor2()
reactor1 = Reactor1()
battery = Battery(50)

#calculate conditions at each hourly state
for i in range(data_length):
    state = Hourly_state(i)
        
    #consumed and stored energy
    generated_kw[i + idle_start_length] = state.wind_power + state.solar_power
    
    r1_e_current, r2_e_current, e_to_battery = distribute_energy(
        generated_kw[i+ idle_start_length], 
        battery.charge, 
        r1_e[i+idle_start_length-alloc_step:i+idle_start_length], 
        r2_e[i+idle_start_length-alloc_step:i+idle_start_length])
    
    battery.charge += e_to_battery*battery.efficiency
    battery_charge[i + idle_start_length] = battery.charge
    consumed_kw[i + idle_start_length] = r1_e_current + r2_e_current
    
    #calculate reactor 1 state
    r1_cos_current = reactor1.react(r1_e_current,r1_prev)
    r1_cos_out[i+idle_start_length] = r1_cos_current
    r1_e[i+idle_start_length] = r1_e_current
    r1_prev = r1_cos_current
    
    #calculate reactor 2 state
    r2_sx_current = reactor2.react(r2_e_current,r2_prev)
    r2_sx_out[i+idle_start_length] = r2_sx_current
    r2_e[i+idle_start_length] = r2_e_current
    r2_prev = r2_sx_current

graph_margin = dict(b=60, l=20, r=20, t=60)

#Reactor 1 output figure
fig_r1 = make_subplots(rows=1,cols=1,
                             specs=[[dict(secondary_y= True)]])
fig_r1.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="COS produced",
                                  legendgroup=1),
                       row=1, col=1, secondary_y=False)
fig_r1.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Energy input",
                                  legendgroup=1),
                       row=1, col=1, secondary_y=True)
fig_r1.update_xaxes(title_text="Hours before present", range=[-50,0],
                          row=1, col=1)
fig_r1.update_yaxes(title_text="Kg COS per hour", range=[0,1], 
                    secondary_y=False, row=1, col=1)
fig_r1.update_yaxes(title_text="kW", range=[0,6], 
                    secondary_y=True, row=1, col=1)
fig_r1.update_layout(title_text="H2S + CO2 => COS + H2O", title_x=0.5,
                     title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01),
                     margin=graph_margin)

#Reactor 2 output figure
fig_r2 = make_subplots(rows=1,cols=1,
                             specs=[[dict(secondary_y= True)]])
fig_r2.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Sx produced",
                                  legendgroup=1),
                     row=1, col=1, secondary_y=False)
fig_r2.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Energy input",
                                  legendgroup=1),
                     row=1, col=1, secondary_y=True)
fig_r2.update_xaxes(title_text="Hours before present", range=[-50,0],
                          row=1, col=1)
fig_r2.update_yaxes(title_text="Kg Sx per hour", range=[0,1], 
                    secondary_y=False, row=1, col=1)
fig_r2.update_yaxes(title_text="kW", range=[0,20], 
                    secondary_y=True, row=1, col=1)
fig_r2.update_layout(title_text="COS => CO + Sx", title_x=0.5,
                     title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01),
                     margin=graph_margin)

#Reactor 1 reaction curve figure
r1_rxn_curve_x = np.linspace(0,10,20)
r1_rxn_curve_y= [reactor1.ss_output(i) for i in r1_rxn_curve_x]

fig_r1_rxn = make_subplots(rows=1,cols=1)
fig_r1_rxn.add_trace(go.Scatter(x=[], y=[], mode= "lines", 
                                name="R1 reaction curve"), row=1, col=1)
fig_r1_rxn.add_trace(go.Scatter(x=[], y=[], marker=dict(color="red", size=20),
                                mode="markers", name="Current state"), row=1, col=1)
fig_r1_rxn.update_xaxes(title_text="kW", range=[0,10],row=1, col=1)
fig_r1_rxn.update_yaxes(title_text="Kg COS per hour", range=[0,1.6],row=1, col=1)
fig_r1_rxn.update_layout(title_text="Steady state COS output versus energy", title_x=0.5,
                         title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                         legend=dict(yanchor="bottom", xanchor="right", y=0.02, x=0.99),
                         margin=graph_margin)

#Reactor 2 reaction curve figure
r2_rxn_curve_x = np.linspace(0,16,20)
r2_rxn_curve_y= [reactor2.ss_output(i) for i in r2_rxn_curve_x]

fig_r2_rxn = make_subplots(rows=1,cols=1)
fig_r2_rxn.add_trace(go.Scatter(x=[], y=[], mode= "lines", 
                                name="R2 reaction curve"), row=1, col=1)
fig_r2_rxn.add_trace(go.Scatter(x=[], y=[], marker=dict(color="red", size=20),
                                mode="markers", name="Current state"), row=1, col=1)
fig_r2_rxn.update_xaxes(title_text="kW", range=[0,16], row=1, col=1)
fig_r2_rxn.update_yaxes(title_text="Kg Sx per hour", range=[0,1.1], row=1, col=1)
fig_r2_rxn.update_layout(title_text="Steady state Sx output versus energy", title_x=0.5,
                         title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                         legend=dict(yanchor="bottom", xanchor="right", y=0.02, x=0.99),
                         margin=graph_margin)

#Energy allocation figure
fig_e_allo = make_subplots(rows=1,cols=1,
                             specs=[[dict(secondary_y= True)]])
fig_e_allo.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Energy produced by renewables",
                                  legendgroup=1),
                       row=1, col=1, secondary_y=False)
fig_e_allo.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Energy allocated to plant",
                                  legendgroup=1),
                       row=1, col=1, secondary_y=False)
fig_e_allo.add_trace(go.Scatter(x=[], y=[], mode= "lines", name="Battery charge",
                                  legendgroup=1),
                       row=1, col=1, secondary_y=True)
fig_e_allo.update_xaxes(title_text="Hours before present", range=[-50,0],
                          row=1, col=1)
fig_e_allo.update_yaxes(title_text="kW produced/consumed", range=[0,20], 
                    secondary_y=False, row=1, col=1)
fig_e_allo.update_yaxes(title_text="kW stored in battery", range=[0,100], 
                    secondary_y=True, row=1, col=1)
fig_e_allo.update_layout(title_text="Energy Allocation", title_x=0.5,
                     title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01),
                     margin=graph_margin)


def update(n_intervals):
    r1_updates = [
        dict(x=[x, x], 
             y=[
                 [r1_cos_out[i] for i in range(n_intervals,n_intervals+51)],
                 [r1_e[i] for i in range(n_intervals,n_intervals+51)]
               ]
             ),
            [0,1], 51, 51]#51 is for 51 points that are replaced each iteration
    r2_updates = [
        dict(x=[x, x], 
             y=[
                 [r2_sx_out[i] for i in range(n_intervals,n_intervals+51)],
                 [r2_e[i] for i in range(n_intervals,n_intervals+51)]
               ]
             ),
            [0,1], 51, 51] 
    r1_rxn_updates = [
        dict(x=[r1_rxn_curve_x, [r1_e[n_intervals+idle_start_length]]*20], 
             y=[
                 r1_rxn_curve_y,
                 [reactor1.ss_output(r1_e[n_intervals+idle_start_length])]*20
               ]
             ),
            [0,1], 20, 1]
    r2_rxn_updates = [
            dict(x=[r2_rxn_curve_x, [r2_e[n_intervals+idle_start_length]]*20], 
                 y=[
                     r2_rxn_curve_y,
                     [reactor2.ss_output(r2_e[n_intervals+idle_start_length])]*20
                   ]
                 ),
                [0,1], 20, 1]
    e_allocation = [
        dict(x=[x, x, x], 
             y=[
                 [generated_kw[i] for i in range(n_intervals,n_intervals+51)],
                 [consumed_kw[i] for i in range(n_intervals,n_intervals+51)],
                 [battery_charge[i] for i in range(n_intervals,n_intervals+51)]
               ]
             ),
            [0,1,2], 51, 51, 51]
    return r1_updates, r2_updates, r1_rxn_updates, r2_rxn_updates, e_allocation