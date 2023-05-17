# -*- coding: utf-8 -*-
"""
Created on Wed May 17 07:53:37 2023

@author: rhanusa
"""
import pandas as pd
#from dashboard_components import pph
pph=10

#sp: solar panel
#wt: wind turbine
sp_efficiency = 0.1
sp_area = 1000 #m^2
wt_cut_in = 13 #km/h
wt_rated_speed = 50 #km/h
wt_cut_out = 100 #km/h
wt_max_energy = 5 #kw
data_length = 200
r1_max = 1
r1_min = 0.05
r2_max = 4
r2_min = 1
battery_max = 100

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
        return (wt_max_energy*(windspeed - wt_cut_in)/(wt_rated_speed - wt_cut_in))

#each system state comprises of a period of 6 minutes (10/hour) because pph = 10
class Hourly_state:
    def __init__(self,hour):
        self.time = df_weather["time"][hour]
        self.wind = df_weather["windspeed_100m (km/h)"][hour] #km/h
        self.wind_energy = calc_wind_energy(self.wind) #kW
        self.wind_power = self.wind_energy*pph #kWh
        self.solar = df_weather["shortwave_radiation (W/m²)"][hour] #W/m2
        self.solar_energy = calc_solar_energy(self.solar) #kW
        self.solar_power = self.solar_energy*pph #kWh
        
        
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