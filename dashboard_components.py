# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:30:04 2023

@author: rhanusa
"""
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from weather_energy_components import data_length
from plant_components import Battery, Reactor1, Reactor2
from weather_energy_components import Hourly_state, distribute_energy, battery_max
import math


#periods per hour. Reactor states will be calculated "pph" times per hour. eg
#if pph = 10, states are calculated every 1hr/10 = 6 min. This is necessary because 
#the weather data hourly, but plant state is calculated more frequently
pph = 10

#step duration for each change of energy allocation
alloc_step = 10

#length of graphs' x axes.
idle_start_length = 500
x = [i*60/pph for i in range(-500,1)] ##changed 50 to 500

num_periods = data_length*pph

r1_cos_out = np.zeros(num_periods + idle_start_length)
r2_sx_out = np.zeros(num_periods + idle_start_length)
r1_e = np.zeros(num_periods + idle_start_length)
r2_e = np.zeros(num_periods + idle_start_length)
generated_kw = np.zeros(num_periods + idle_start_length)
consumed_kw = np.zeros(num_periods + idle_start_length)
battery_charge = [50]*(num_periods + idle_start_length)
kw_to_battery_arr = np.zeros(num_periods + idle_start_length)

r1_prev = 0 
r2_prev = 0 
reactor2 = Reactor2()
reactor1 = Reactor1()
battery = Battery(50)

#calculate conditions at each hourly state
for hour in range(data_length):
    state = Hourly_state(hour)
    for i in range(pph):
        period = hour*pph + i
            
        #consumed and stored energy
        generated_kw[period + idle_start_length] = state.wind_power/pph + state.solar_power/pph
        
        r1_e_current, r2_e_current, e_to_battery = distribute_energy(
            generated_kw[period+ idle_start_length], 
            battery.charge, 
            r1_e[period + idle_start_length-alloc_step:period + idle_start_length], 
            r2_e[period + idle_start_length-alloc_step:period + idle_start_length])
        
        battery.charge += e_to_battery * battery.efficiency / pph #divide by pph because this is kWh
        battery_charge[period + idle_start_length] = battery.charge
        consumed_kw[period + idle_start_length] = r1_e_current + r2_e_current
        kw_to_battery_arr[period + idle_start_length] = e_to_battery
        
        #calculate reactor 1 state
        r1_cos_current = reactor1.react(r1_e_current, r1_prev)
        r1_cos_out[period + idle_start_length] = r1_cos_current
        r1_e[period + idle_start_length] = r1_e_current
        r1_prev = r1_cos_current
        
        #calculate reactor 2 state
        r2_sx_current = reactor2.react(r2_e_current, r2_prev)
        r2_sx_out[period + idle_start_length] = r2_sx_current
        r2_e[period + idle_start_length] = r2_e_current
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
fig_r1.update_xaxes(title_text="Minutes before present", range=[-500,0],
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
fig_r2.update_xaxes(title_text="Minutes before present", range=[-500,0],
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
fig_e_allo.update_xaxes(title_text="Minutes before present", range=[-500,0],
                          row=1, col=1)
fig_e_allo.update_yaxes(title_text="kW produced/consumed", range=[0,20], 
                    secondary_y=False, row=1, col=1)
fig_e_allo.update_yaxes(title_text="kWh stored in battery", range=[0,100], 
                    secondary_y=True, row=1, col=1)
fig_e_allo.update_layout(title_text="Energy Allocation", title_x=0.5,
                     title=dict(yref="paper", y=1, yanchor="bottom", pad=dict(b=20)),
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01),
                     margin=graph_margin)

#bar chart for battery level
fig_bat_lvl = make_subplots(rows=1,cols=1)
fig_bat_lvl.add_trace(go.Bar(x=[], y=[]))

def update1(n_intervals):
    r1_updates = [
        dict(x=[x, x], 
             y=[
                 [r1_cos_out[i] for i in range(n_intervals,n_intervals + idle_start_length + 1)],
                 [r1_e[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)]
               ]
             ),
            [0,1], idle_start_length + 1, idle_start_length + 1]#51 is for 51 points that are replaced each iteration
    r2_updates = [
        dict(x=[x, x], 
             y=[
                 [r2_sx_out[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)],
                 [r2_e[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)]
               ]
             ),
            [0,1], idle_start_length + 1, idle_start_length + 1] 
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
                 [generated_kw[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)],
                 [consumed_kw[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)],
                 [battery_charge[i] for i in range(n_intervals,n_intervals+idle_start_length + 1)]
               ]
             ),
            [0,1,2], idle_start_length + 1, idle_start_length + 1, idle_start_length + 1]
    
    kw_to_battery = kw_to_battery_arr[n_intervals + idle_start_length]
    
    state = Hourly_state(math.floor(n_intervals/pph))
    time = state.time
    
    kw_gen = generated_kw[n_intervals + idle_start_length]
    
    bat_lvl_full = battery_charge[n_intervals + idle_start_length]
    bat_lvl_empty = battery_max - battery_charge[n_intervals + idle_start_length]
 
    fig_bat_lvl = go.Figure(data=[go.Bar(x=[1], y=[bat_lvl_full], marker_color="green"),
                                  go.Bar(x=[1], y=[bat_lvl_empty],  marker_color="gray")])
    fig_bat_lvl.update_layout(title_text="Battery Level", title_x=0.5, title_y=0.96,
                              barmode='stack', paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor='rgba(0, 0, 0, 0)', showlegend=False,
                              margin=dict(l=20, r=20, t=20, b=20), height=200,
                              width=150)
    fig_bat_lvl.update_xaxes(showticklabels=False)
    fig_bat_lvl.update_yaxes(tickmode="array", tickvals=[round(bat_lvl_full)])


    return (r1_updates, r2_updates, r1_rxn_updates, r2_rxn_updates, e_allocation,
            round(kw_to_battery,2), time.strftime("%d-%m-%Y %H:%M:%S"), fig_bat_lvl,
            round(kw_gen,2), round(r1_e[n_intervals + idle_start_length],2),
            round(r2_e[n_intervals + idle_start_length],2),"X.XX")

#update function for r1 reactor saturations

def update_r1s(n_intervals):
    return