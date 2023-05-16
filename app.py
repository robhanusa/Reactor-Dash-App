# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:43:06 2023

@author: rhanusa
"""
import numpy as np
import math
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#sp: solar panel
#wt: wind turbine
sp_efficiency = 0.1
sp_area = 1000 #m^2
wt_cut_in = 13 #km/h
wt_rated_speed = 50 #km/h
wt_cut_out = 100 #km/h
wt_max_energy = 5 #kw
data_length = 200
r1_cleaning_speed = 0.1
r1_max = 1
r1_min = 0.05
r2_max = 4
r2_min = 1
battery_max = 100

#length of time before the energy allocation to reactors is allowed to change
alloc_step = 10

cols = ["time","windspeed_100m (km/h)","shortwave_radiation (W/m²)"]

df_weather = pd.read_csv("wind_solar_2013-2022_open-meteo.com.csv",
                         skiprows=3,
                         usecols=cols,
                         nrows=data_length)

df_weather["time"] = pd.to_datetime(df_weather["time"])

#calc kw's generated from solar panels
def calc_solar_energy(solar_radiation):
    return solar_radiation*sp_area*sp_efficiency/1000 

#calc kw's generated from wind turbines
#for simplicity, assuming linear relationship between cut-in and rated speeds
#for complex relationships, look at (Sohoni 2016)
def calc_wind_energy(windspeed):
    if windspeed < wt_cut_in or windspeed > wt_cut_out: 
        return 0
    elif windspeed > wt_rated_speed: 
        return wt_max_energy
    else:
        return wt_max_energy*(windspeed - wt_cut_in)/(wt_rated_speed - wt_cut_in)

#each system state comprises of a period of 6 minutes (10/hour)
class Hourly_state:
    def __init__(self,hour):
        self.time = df_weather["time"][hour]
        self.wind = df_weather["windspeed_100m (km/h)"][hour] #km/h
        self.wind_energy = calc_wind_energy(self.wind) #kW
        self.wind_power = self.wind_energy #kWh
        self.solar = df_weather["shortwave_radiation (W/m²)"][hour] #W/m2
        self.solar_energy = calc_solar_energy(self.solar) #kW
        self.solar_power = self.solar_energy #kWh
        
class Battery:
    def __init__(self,charge):
        self.charge = charge
        self.efficiency = 0.9 #note this might be a function of charge etc, so not a simple constant
        
class Reactor1:
    ku = .1 
    kd = .2 
    
    def __init__(self):
        self.state = "idle"
        self.saturation = 0 

    @classmethod
    def ss_output(cls, energy):
        return 2/(1 + math.exp(-energy + 1))-0.54
    
    def react(cls, energy, prev):
        Reactor1.state = "active"
        e_t = Reactor1.ss_output(energy)-prev
        if e_t > 0:
            cos_produced = prev + Reactor1.ku*e_t
        else:
            cos_produced = prev + Reactor1.kd*e_t
        Reactor1.add_water(cos_produced)
        return cos_produced
        
    def add_water(cos_produced):
        #Reactor1.saturation += cos_produced/1000
        Reactor1.state = "active"
        
    def check_saturation():
        if Reactor1.saturation < 1:
            return False
        else:
            return True

    def clean():
        if Reactor1.state == "idle":
            return;
        else:
            Reactor1.saturation = max(0, Reactor1.saturation - r1_cleaning_speed)
        if Reactor1.saturation == 0: 
            Reactor1.state == "idle"
        else:
            Reactor1.state == "cleaning"
            
class Reactor2: 
    ku = .1 
    kd = .2 
    
    def __init__(self):
        self.state = "idle"
        
    @classmethod
    def ss_output(cls, energy):
        return 1/(1 + math.exp(-energy + 3))-0.05
    
    def react(cls, energy, prev):
        Reactor2.state = "active"
        e_t = Reactor2.ss_output(energy)-prev
        if e_t > 0:
            return prev + Reactor2.ku*e_t
        else:
            return prev + Reactor2.kd*e_t

#need to adjust this so that the COS produced = COS consumed
#now this is only a theoretical process control logic
def distribute_energy(energy_generated, energy_stored, prev10_r1, prev10_r2): 
    if len(set(prev10_r1)) > 1:
        to_r1 = prev10_r1[-1]
        to_r2 = prev10_r2[-1]
    elif energy_stored + energy_generated < r2_min + r1_min or energy_stored/battery_max < 0.10:
        to_r1 = 0
        to_r2 = 0
    elif energy_stored/battery_max < 0.30:
        to_r1 = 0.5
        to_r2 = 1
    elif energy_stored/battery_max < 0.50:
        to_r1 = .7
        to_r2 = 2
    else:
        to_r1 = r1_max
        to_r2 = r2_max
        
    to_battery = energy_generated - to_r1 - to_r2

    return to_r1, to_r2, to_battery
            
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
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01))

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
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01))

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
                         legend=dict(yanchor="bottom", xanchor="right", y=0.02, x=0.99))

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
                         legend=dict(yanchor="bottom", xanchor="right", y=0.02, x=0.99))

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
                     legend=dict(yanchor="top", xanchor="left", y=0.99, x=0.01))


#%%
            
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)
app.layout = dbc.Container([
    dbc.Row([dbc.Col(dcc.Graph(id="reactor1_output_graph", figure=fig_r1), width=6),
            dbc.Col(dcc.Graph(id="reactor2_output_graph", figure=fig_r2), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="r1_rxn_graph", figure=fig_r1_rxn), width=6),
             dbc.Col(dcc.Graph(id="r2_rxn_graph", figure=fig_r2_rxn), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="energy_allocation_graph",figure=fig_e_allo))]),
    dcc.Interval(id="interval",interval=200, 
                  n_intervals=0, max_intervals=len(r2_sx_out)-51)])

@app.callback(Output(component_id='reactor1_output_graph', component_property='extendData'),
              Output(component_id='reactor2_output_graph', component_property='extendData'),
              Output(component_id='r1_rxn_graph', component_property='extendData'),
              Output(component_id='r2_rxn_graph', component_property='extendData'),
              Output(component_id='energy_allocation_graph', component_property='extendData'),
              Input(component_id='interval', component_property='n_intervals'))

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

if __name__ == "__main__":
    app.run_server(debug=True)