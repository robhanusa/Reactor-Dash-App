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
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

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
        
class Reactor1:
    ku = .1 
    kd = .2 
    
    def __init__(self):
        self.state = "idle"
        self.saturation = 0 

    @classmethod
    def ss_output(cls, energy):
        return 1/(1 + math.exp(-energy + 1))
    
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
        return 1/(1 + math.exp(-energy + 3))
    
    def react(cls, energy, prev):
        Reactor2.state = "active"
        e_t = Reactor2.ss_output(energy)-prev
        if e_t > 0:
            return prev + Reactor2.ku*e_t
        else:
            return prev + Reactor2.kd*e_t
            
            
idle_start_length = 50
x = [i for i in range(-50,1)]
r1_cos_out = [0]*(data_length+idle_start_length)
r2_sx_out = [0]*(data_length+idle_start_length)
r1_e = [0]*(data_length+idle_start_length)
r2_e = [0]*(data_length+idle_start_length)

r1_prev = 0 
r2_prev = 0 
reactor2 = Reactor2()
reactor1 = Reactor1()

#calculate conditions at each hourly state
for i in range(data_length):
    state = Hourly_state(i)
    
    #calculate reactor 1 state
    r1_e_current = (state.wind_power + state.solar_power)*.25 #the .75 is an arbitrary factor for distributing the energy.
    r1_cos_current = reactor1.react(r1_e_current,r1_prev)
    r1_cos_out[i+idle_start_length] = r1_cos_current
    r1_e[i+idle_start_length] = r1_e_current
    r1_prev = r1_cos_current
    
    #calculate reactor 2 state
    r2_e_current = (state.wind_power + state.solar_power)*.75 #the .75 is an arbitrary factor for distributing the energy.
    r2_sx_current = reactor2.react(r2_e_current,r2_prev)
    r2_sx_out[i+idle_start_length] = r2_sx_current
    r2_e[i+idle_start_length] = r2_e_current
    r2_prev = r2_sx_current
    
#https://dash.plotly.com/dash-core-components/graph#using-the-low-level-interface-with-dicts-&-lists
fig_r1_out = go.Figure()
fig_r1_out.update_layout(title=go.layout.Title(text="H2S + CO2 => COS + H2O"),
                         title_x=0.5)
fig_r1_out.add_trace(go.Scatter(x=[],y=[], mode= "lines", name="COS produced"))
fig_r1_out.add_trace(go.Scatter(x=[],y=[], mode= "lines", name="Energy input"))
fig_r1_out.update_xaxes(range=[-50,0], title_text="Hours before present")
fig_r1_out.update_yaxes(range=[0,1])

fig_r2_out = go.Figure()
fig_r2_out.add_trace(go.Scatter(x=[],y=[], mode= "lines", name="Sx produced"))
fig_r2_out.add_trace(go.Scatter(x=[],y=[], mode= "lines", name="Energy input"))
fig_r2_out.update_layout(title=go.layout.Title(text="COS => CO + Sx"),
                         title_x=0.5)
fig_r2_out.update_xaxes(range=[-50,0], title_text="Hours before present")
fig_r2_out.update_yaxes(range=[0,1])

#%%
            
app = dash.Dash(__name__, update_title=None)
app.layout = html.Div([
    dcc.Graph(id="r1_out_graph", figure=fig_r1_out),
    dcc.Graph(id="r2_out_graph", figure=fig_r2_out),
    dcc.Interval(id="interval",interval=100, 
                 n_intervals=0, max_intervals=len(r2_sx_out)-51)])

#example of updating entire figure: https://dash.plotly.com/basic-callbacks
#notice "transition_duration" i can use to make smooth transitions

@app.callback(Output(component_id='r1_out_graph', component_property='extendData'),
              Output(component_id='r2_out_graph', component_property='extendData'),
              Input(component_id='interval', component_property='n_intervals'))

def update(n_intervals):
    r1_out_graph_updates = [
        dict(x=[x, x], 
             y=[
                 [r1_cos_out[i] for i in range(n_intervals,n_intervals+51)],
                 [r1_e[i] for i in range(n_intervals,n_intervals+51)]
               ]
             ),
            [0,1], 51, 51] #51 is for 51 points that are replaced each iteration
    
    r2_out_graph_updates = [
        dict(x=[x, x], 
             y=[
                 [r2_sx_out[i] for i in range(n_intervals,n_intervals+51)],
                 [r2_e[i] for i in range(n_intervals,n_intervals+51)]
                 ]
             ), 
            [0,1], 51, 51]
    return r1_out_graph_updates, r2_out_graph_updates


if __name__ == "__main__":
    app.run_server(debug=True)