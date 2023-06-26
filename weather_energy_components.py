# -*- coding: utf-8 -*-
"""
Created on Wed May 17 07:53:37 2023

@author: rhanusa
"""
import pandas as pd
from plant_components import pph

# sp: solar panel
# wt: wind turbine
sp_efficiency = 0.1
sp_area = 1000 # m^2
wt_cut_in = 13 # km/h
wt_rated_speed = 50 # km/h
wt_cut_out = 100 # km/h
wt_max_energy = 5 # kW
data_length = 87671
battery_max = 20 #kWh

cols = ["time","windspeed_100m (km/h)","shortwave_radiation (W/m²)"]

df_weather = pd.read_csv("wind_solar_2013-2022_open-meteo.com.csv",
                         skiprows=3,
                         usecols=cols,
                         nrows=data_length)

df_weather["time"] = pd.to_datetime(df_weather["time"])

# calc kw's generated from solar panels
def calc_solar_energy(solar_radiation):
    return solar_radiation*sp_area*sp_efficiency/1000

# calc kw's generated from wind turbines
# for simplicity, assuming linear relationship between cut-in and rated speeds
# for complex relationships, look at (Sohoni 2016)
def calc_wind_energy(windspeed):
    if windspeed < wt_cut_in or windspeed > wt_cut_out: 
        return 0
    elif windspeed > wt_rated_speed: 
        return wt_max_energy
    else:
        return (wt_max_energy*(windspeed - wt_cut_in)/(wt_rated_speed - wt_cut_in))

# each system state comprises of a period of 6 minutes (10/hour) when pph = 10
class Hourly_state:
    def __init__(self,hour):
        self.time = df_weather["time"][hour]
        self.wind = df_weather["windspeed_100m (km/h)"][hour] #km/h
        self.wind_energy = calc_wind_energy(self.wind) #kW
        self.wind_power = self.wind_energy*pph #kWh
        self.solar = df_weather["shortwave_radiation (W/m²)"][hour] #W/m2
        self.solar_energy = calc_solar_energy(self.solar) #kW
        self.solar_power = self.solar_energy*pph #kWh
        self.month = df_weather["time"][hour].month
        self.hour_of_day = df_weather["time"][hour].hour
        
class Energy_flow():
    def __init__(self):
        self.to_r1 = 0
        self.to_r2 = 0 
        self.to_condenser = 0 
        self.to_battery = 0
        self.from_grid = 0
        

    