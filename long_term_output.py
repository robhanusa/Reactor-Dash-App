# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:47:37 2023

@author: rhanusa
"""

''' 
outputs:
    (values)
        Total energy from grid (eventually cost from grid, if we account for tiered pricing)
        Sx produced
        Sx filter changeovers
        r1 changeovers
        
    (Adjustable inputs)
        Capex (how large the battery, how many wind turbines, area of solar panels)
    
    (data for graphs)
        Energy from grid, averaged for each hour, grouped by month
        Sx produced per month
        
Target production rate is 30 mol S per hr, on average.
This comes to 262.8 kmol S per year
(note that the official goal is 239.7 kmol S / year [266.4 kmol COS conversion @ 90% efficiency],
 but general assumption of uptime is 8000 hours per year. This model doesn't [yet]
 account for downtime, so we're currently targeting 262.8 kmol S / year)
'''
import numpy as np
import plant_components as pc
from plant_components import pph
import weather_energy_components as wec
import time
import input_specs as ins
import pandas as pd

start2 = time.time()

# generate matrix of inputs
wt_list = [0, 1] # number of 1MW wind turbines
sp_list = [5000, 10000, 15000] # area in m2 of solar panels
b_list = [516, 1144, 2288] # battery sizes in kW
c1_list = [0, 0.05, 0.1] # constants for r2_max eqn
c2_list = [0, 0.05, 0.1]
c3_list = [0, 2*10**(-5), 4*10**(-5)]

# forecast = np.zeros([wec.data_length-12,12])
# for hour in range(wec.data_length-12):
#     state = wec.Hourly_state(hour, ins.solar_panel_specs, ins.wind_turbine_specs)

#     # Forecast data
#     for j in range(12):
#         future_state = wec.Hourly_state(hour+j+1, ins.solar_panel_specs, ins.wind_turbine_specs)
#         forecast[hour][j] = wec.calc_generated_kw(future_state)


# Pre-calculate forecast data

forecast_store = np.zeros([len(wt_list), len(sp_list), wec.data_length-12, 2])
forecast_arr = np.zeros(12)

for i in range(len(wt_list)):
    wt = wt_list[i]
    wind_turbine_specs = {
        "cut_in": 13, # km/h
        "rated_speed": 50, # km/h
        "cut_out": 100, # km/h
        "max_energy": 1000, # kW
        "count": wt,
        "cost": wt*1.5*10**6 # EUR/MW 
        }
    
    for j in range(len(sp_list)):
        sp = sp_list[j]
        solar_panel_specs = {
            "area": sp, # m^2
            "efficiency": 0.1,
            "cost": sp*200/1.1 # $200/m2 (/1.1 eur to usd)
            }

        for hour in range(wec.data_length-12):
            for k in range(12): 
                future_state = wec.Hourly_state(hour+k+1, solar_panel_specs, wind_turbine_specs)
                forecast_arr[k] = wec.calc_generated_kw(future_state)
            
                forecast_store[i][j][hour][0] = sum(forecast_arr[0:6])
                forecast_store[i][j][hour][1] = sum(forecast_arr[6:13])
                
end = time.time()
print("Time elapsed: ", end - start2)

#%% Run DOE scenarios to generate profit at different conditions

def run_scenario(wt_level, sp_level, b_level, c1_level, c2_level, c3_level):
    
    start = time.time()
    
    wt = wt_list[wt_level]
    sp = sp_list[sp_level]
    b = b_list[b_level]
    c1 = c1_list[c1_level]
    c2 = c2_list[c2_level]
    c3 = c3_list[c3_level]
    
    battery_specs = { 
        "max_charge": b, # kWh
        "cost": 1000*b
        }

    solar_panel_specs = {
        "area": sp, # m^2
        "efficiency": 0.1,
        "cost": sp*200/1.1 # $200/m2 (/1.1 eur to usd)
        }

    wind_turbine_specs = {
        "cut_in": 13, # km/h
        "rated_speed": 50, # km/h
        "cut_out": 100, # km/h
        "max_energy": 1000, # kW
        "count": wt,
        "cost": wt*1.5*10**6 # EUR/MW 
        }

    r2_max_constants = {
        "c1": c1,
        "c2": c2,
        "c3": c3
        }
    
    # initialize counters
    e_from_grid = 0 # kwh
    e_from_grid_hourly = np.zeros([24,12]) # kw per hour per month
    e_to_grid = 0 # kwh
    e_to_grid_hourly = np.zeros([24,12]) # kw per hour per month
    total_renewable = 0 # kwh
    total_renewable_hourly = np.zeros([24,12]) # kw per hour per month
    total_sx = 0 # mol
    total_sx_monthly = np.zeros(12)
    # hour_tally = np.zeros([24,12])
    # month_tally = np.zeros(12)
    
    # Create arrays to store all calculated data that will be displayed.
    # This helps minimize the calculations occuring in the callback function and
    # lets the dashboard update more smoothly
    # sx_sat = 0
    # sx_filter_tally = 0
    # r1_changeovers_tally = 0
    
    # Initiate necessary variables
    r2_prev = 0 
    reactor1_1 = pc.Reactor1()
    # reactor1_2 = pc.Reactor1()
    reactor1_1.state = "active"
    reactor2 = pc.Reactor2()
    # sx_filter = pc.Sx_filter()
    battery = pc.Battery(0.5*battery_specs["max_charge"], battery_specs)
    energy_flow = wec.Energy_flow()
    energy_tally = pph
    r2_e_prev = 0
    # prev_month = 0
    # prev_hour = 0
    
    # forecast_arr = np.zeros(12)
    
    # Calculate conditions at each hourly state and store in arrays
    for hour in range(wec.data_length-12):
        state = wec.Hourly_state(hour, ins.solar_panel_specs, ins.wind_turbine_specs)
        
        # Energy flowing to the plant
        energy_generated = wec.calc_generated_kw(state)
        total_renewable += energy_generated
        total_renewable_hourly[state.hour_of_day][state.month-1] += energy_generated
        
        # # Forecast data
        # for j in range(12):
        #     future_state = wec.Hourly_state(hour+j+1, solar_panel_specs, wind_turbine_specs)
        #     forecast_arr[j] = wec.calc_generated_kw(future_state)
        
        # forecast = (sum(forecast_arr[0:6]), sum(forecast_arr[6:13]))
        
        forecast = (forecast_store[wt_level][sp_level][hour][0],
                    forecast_store[wt_level][sp_level][hour][1])
        
        # Allow for multiple periods per hour
        for i in range(pph):
            
            # Energy distribution for current period
            energy_tally, r2_e_prev = wec.distribute_energy(energy_generated, 
                                                            energy_tally, 
                                                            r2_e_prev, 
                                                            energy_flow, 
                                                            battery, 
                                                            reactor2,
                                                            r2_max_constants, 
                                                            forecast)
            
            # Update battery charge
            battery.charge += wec.battery_charge_differential(energy_flow.to_battery, battery)
            
            # Calculate reactor 2 state
            r2_sx_current = reactor2.react(energy_flow.to_r2, r2_prev)
            total_sx += r2_sx_current/pph
            r2_prev = r2_sx_current
            total_sx_monthly[state.month-1] += r2_sx_current/pph
            
            # # Calculate reactor 1 states and tally changeovers
            # r1_changeovers_tally = pc.update_reactor_1(r2_sx_current, r1_changeovers_tally, reactor1_1, reactor1_2)
            
            # # Update Sx filter saturation and tally filter changes
            # sx_sat, sx_filter_tally = pc.update_sx_filter(sx_sat, sx_filter_tally, sx_filter, r2_sx_current)
            
            # Add up energy taken from grid
            if energy_flow.from_grid > 0:
                e_from_grid += energy_flow.from_grid/pph
                e_from_grid_hourly[state.hour_of_day][state.month-1] += energy_flow.from_grid/pph
                
            if energy_flow.from_grid < 0:
                e_to_grid -= energy_flow.from_grid/pph
                e_to_grid_hourly[state.hour_of_day][state.month-1] -= energy_flow.from_grid/pph
            
            # if prev_hour != state.hour_of_day: hour_tally[state.hour_of_day][state.month-1] += 1
            # if prev_month != state.month: month_tally[state.month-1] += 1
            
            # prev_hour = state.hour_of_day
            # prev_month = state.month
    
    # ave_renewable_hourly = total_renewable_hourly / hour_tally
    # ave_from_grid_hourly = e_from_grid_hourly / hour_tally
    # ave_to_grid_hourly = e_to_grid_hourly / hour_tally
    # ave_sx_monthly = total_sx_monthly / month_tally
    
    revenue = 9.6*total_sx + 0.1*e_to_grid
    opex = 0.25*e_from_grid
    capex = battery_specs["cost"] + solar_panel_specs["cost"] + wind_turbine_specs["cost"]
    profit = revenue - opex - capex
    
    end = time.time()
    print("Time elapsed: ", end - start)
    
    return profit, revenue, opex, capex


# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe = pd.read_excel("DOE.xlsx")

results_matrix = np.zeros([len(doe), 10])

doe[["profit", "revenue", "opex", "capex"]] = np.zeros([len(doe), 4])

for run in range(len(doe)):
    
    print("Run: ", run)
    
    wt_level = doe["wt_level"][run]
    sp_level = doe["sp_level"][run]
    b_level = doe["b_level"][run]
    c1_level = doe["c1_level"][run]
    c2_level = doe["c2_level"][run] 
    c3_level = doe["c3_level"][run]
    
    profit, revenue, opex, capex = run_scenario(wt_level, 
                                                sp_level, 
                                                b_level, 
                                                c1_level, 
                                                c2_level, 
                                                c3_level)
    doe["profit"][run] = profit
    doe["revenue"][run] = revenue
    doe["opex"][run] = opex
    doe["capex"][run] = capex
    
    # doe.loc[run, "profit"] = profit
    # doe.loc[run, "revenue"] = revenue
    # doe.loc[run, "opex"] = opex
    # doe.loc[run, "capex"] = capex

end2 = time.time()    
print("Time elapsed: ", end2 - start2)

#%% Generate a model for profit as a function of the input parameters



#%% Optimization: find parameter values that maximize profit

