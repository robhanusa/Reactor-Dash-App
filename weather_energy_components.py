# -*- coding: utf-8 -*-
"""
Created on Wed May 17 07:53:37 2023

@author: rhanusa
"""
import pandas as pd
from plant_components import pph

# sp: solar panel
# wt: wind turbine
# sp_efficiency = 0.1
# sp_area = 10000 # m^2
# wt_cut_in = 13 # km/h
# wt_rated_speed = 50 # km/h
# wt_cut_out = 100 # km/h
# wt_max_energy = 1000 # kW
data_length = 87671
battery_max = 1144 #kWh

cols = ["time","windspeed_100m (km/h)","shortwave_radiation (W/m²)"]

df_weather = pd.read_csv("wind_solar_2013-2022_open-meteo.com.csv",
                         skiprows=3,
                         usecols=cols,
                         nrows=data_length)

df_weather["time"] = pd.to_datetime(df_weather["time"])

# calc kw's generated from solar panels
def calc_solar_energy(solar_radiation, solar_panel_specs):
    sp_area = solar_panel_specs['area']
    sp_efficiency = solar_panel_specs['efficiency']
    return solar_radiation*sp_area*sp_efficiency/1000

# calc kw's generated from wind turbines
# for simplicity, assuming linear relationship between cut-in and rated speeds
# for complex relationships, look at (Sohoni 2016)
def calc_wind_energy(windspeed, wind_turbine_specs):
    wt_cut_in = wind_turbine_specs['cut_in']
    wt_cut_out = wind_turbine_specs['cut_out']
    wt_rated_speed = wind_turbine_specs['rated_speed']
    wt_max_energy = wind_turbine_specs['max_energy']
    wt_number = wind_turbine_specs['count']
    if windspeed < wt_cut_in or windspeed > wt_cut_out: 
        return 0
    elif windspeed > wt_rated_speed: 
        return wt_number*wt_max_energy
    else:
        return wt_number*(wt_max_energy*(windspeed - wt_cut_in)/(wt_rated_speed - wt_cut_in))

# each system state comprises of a period of 6 minutes (10/hour) when pph = 10
class Hourly_state:
    def __init__(self,hour, solar_panel_specs, wind_turbine_specs):
        self.time = df_weather["time"][hour]
        self.wind = df_weather["windspeed_100m (km/h)"][hour] #km/h
        self.wind_energy = calc_wind_energy(self.wind, wind_turbine_specs) #kW
        self.wind_power = self.wind_energy*pph #kWh
        self.solar = df_weather["shortwave_radiation (W/m²)"][hour] #W/m2
        self.solar_energy = calc_solar_energy(self.solar, solar_panel_specs) #kW
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
        
def calc_generated_kw(state):
    return state.wind_power/pph + state.solar_power/pph

def allocate_e_to_condenser(to_r2, reactor2):
    condenser_constant = 0.5
    r2_sx_ss = reactor2.ss_output(to_r2)
    to_condenser = condenser_constant * r2_sx_ss
    return to_condenser

# Need to adjust so COS produced = COS consumed
def distribute_energy(energy_generated, energy_tally, r2_e_prev, energy_flow, battery, reactor2):
    energy_stored = battery.charge
    r1_max = 50
    r1_min = 50
    r2_max = 500
    r2_min = 50
    
    # Check if there has already been a change in energy distribution in the last
    # hour. If so, don't change current distribution.
    if energy_tally < pph : 
        energy_tally += 1
    elif energy_stored + energy_generated < r2_min + r1_min:
        energy_flow.to_r1 = r1_min
        energy_flow.to_r2 = r2_min
        energy_flow.to_condenser = allocate_e_to_condenser(energy_flow.to_r2, reactor2)
        energy_flow.from_grid = r2_min + r1_min - (energy_stored + energy_generated)
    elif energy_stored/battery_max < 0.30:
        energy_flow.to_r1 = r1_min
        energy_flow.to_r2 = r2_min
        energy_flow.to_condenser = allocate_e_to_condenser(energy_flow.to_r2, reactor2)
        energy_flow.from_grid = 0
    elif energy_stored/battery_max < 0.50:
        energy_flow.to_r1 = r1_max
        energy_flow.to_r2 = 200
        energy_flow.to_condenser = allocate_e_to_condenser(energy_flow.to_r2, reactor2)
        energy_flow.from_grid = 0
    else:
        energy_flow.to_r1 = r1_max
        energy_flow.to_r2 = r2_max
        energy_flow.to_condenser = allocate_e_to_condenser(energy_flow.to_r2, reactor2)
        energy_flow.from_grid = 0
    
    if r2_e_prev != energy_flow.to_r2: energy_tally = 1
    
    r2_e_prev = energy_flow.to_r2
        
    if energy_flow.from_grid == 0:
        energy_flow.to_battery = energy_generated \
            - energy_flow.to_r1 - energy_flow.to_r2 - energy_flow.to_condenser
    else:
        if energy_stored + energy_generated < r2_min + r1_min:
            energy_flow.to_battery = -energy_stored
        else:
            energy_flow.to_battery = 0

    return energy_tally, r2_e_prev

def battery_charge_differential(e_to_battery, battery):
    if e_to_battery > 0:
        if e_to_battery * battery.efficiency / pph + battery.charge > battery_max:
            return battery_max - battery.charge
        else:
            return e_to_battery * battery.efficiency / pph
    else:
        return e_to_battery / pph