# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:43:06 2023

@author: rhanusa
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_html_components as html
import dash_core_components as dcc

#sp: solar panel
#wt: wind turbine
sp_efficiency = 0.1
sp_area = 1000 #m^2
wt_cut_in = 13 #km/h
wt_rated_speed = 50 #km/h
wt_cut_out = 100 #km/h
wt_max_energy = 5 #kw

r1_cleaning_speed = 0.1

cols = ["time","windspeed_100m (km/h)","shortwave_radiation (W/m²)"]

df_weather = pd.read_csv("wind_solar_2013-2022_open-meteo.com.csv",
                         skiprows=3,
                         usecols=cols,
                         nrows=100)

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
        self.wind_power = self.wind_energy/10 #kWh
        self.solar = df_weather["shortwave_radiation (W/m²)"][hour] #W/m2
        self.solar_energy = calc_solar_energy(self.solar) #kW
        self.solar_power = self.solar_energy/10 #kWh
        
class Reactor1:
    ku = .1 
    kd = .2 
    
    def __init__(self):
        self.state = "idle"
        self.saturation = 0 

    @classmethod
    def ss_output(cls, energy):
        return 1/(1 + math.exp(-.85 * energy + 1))
    
    def react(cls, energy, prev):
        e_t = Reactor1.ss_output(energy)-prev
        if e_t > 0:
            return prev + Reactor1.ku*e_t
        else:
            return prev + Reactor1.kd*e_t
        
    def add_water(cos_produced):
        Reactor1.saturation += cos_produced/1000
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
        return 1/(1 + math.exp(-.85 * energy + 3))
    
    def react(cls, energy, prev):
        if energy == 0:
            Reactor2.state = "idle"
            return 0 
        else: 
            Reactor2.state = "active"
            e_t = Reactor2.ss_output(energy)-prev
            if e_t > 0:
                return prev + Reactor2.ku*e_t
            else:
                return prev + Reactor2.kd*e_t
            
x = [i for i in range(len(df_weather))]
y = [0]*len(df_weather)

for i in range(len(y)):
    state = Hourly_state(i)
    reactor2 = Reactor2()
    total_e = state.wind_power + state.solar_power
    if i == 0:
        r2_current = reactor2.react(total_e,0)
    else:
        r2_current = reactor2.react(total_e,r2_prev)
    y[i] = r2_current
    r2_prev = r2_current
    
            
app = dash.Dash(__name__, update_title=None)
app.layout = html.Div(
    dcc.Graph(
        figure = {
            "data":  [
                {
                    "x":x,
                    "y":y,
                    "type": "lines",
                    }]})
    )

if __name__ == "__main__":
    app.run_server(debug=True)