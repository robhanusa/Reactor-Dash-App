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

start = time.time()

# generate matrix of inputs
wt_list = [0, 1] # number of 1MW wind turbines
sp_list = [5000, 10000, 15,000] # area in m2 of solar panels
b_list = [516, 1144, 2288] # battery sizes in kW
c1_list = [0, 0.05, 0.1] # constants for r2_max eqn
c2_list = [0, 0.05, 0.1]
c3_list = [0, 2*10**(-5), 4*10**(-5)]

forecast = np.zeros([wec.data_length-12,12])
for hour in range(wec.data_length-12):
    state = wec.Hourly_state(hour, ins.solar_panel_specs, ins.wind_turbine_specs)

    # Forecast data
    for j in range(12):
        future_state = wec.Hourly_state(hour+j+1, ins.solar_panel_specs, ins.wind_turbine_specs)
        forecast[hour][j] = wec.calc_generated_kw(future_state)
        
wt = 1
sp = 12222
b = 1111
c1 = .1
c2 = .1 
c3 = .00001

start = time.time()

def simulate(wt, sp, b, c1, c2, c3):
    # initialize counters
    e_from_grid = 0 # kwh
    e_from_grid_hourly = np.zeros([24,12]) # kw per hour per month
    e_to_grid = 0 # kwh
    e_to_grid_hourly = np.zeros([24,12]) # kw per hour per month
    total_renewable = 0 # kwh
    total_renewable_hourly = np.zeros([24,12]) # kw per hour per month
    total_sx = 0 # mol
    total_sx_monthly = np.zeros(12)
    hour_tally = np.zeros([24,12])
    month_tally = np.zeros(12)
    
    # Create arrays to store all calculated data that will be displayed.
    # This helps minimize the calculations occuring in the callback function and
    # lets the dashboard update more smoothly
    sx_sat = 0
    sx_filter_tally = 0
    r1_changeovers_tally = 0
    
    # Initiate necessary variables
    r2_prev = 0 
    reactor1_1 = pc.Reactor1()
    reactor1_2 = pc.Reactor1()
    reactor1_1.state = "active"
    reactor2 = pc.Reactor2()
    sx_filter = pc.Sx_filter()
    battery = pc.Battery(0.5*wec.battery_max, wec.battery_max, ins.battery_specs)
    energy_flow = wec.Energy_flow()
    energy_tally = pph
    r2_e_prev = 0
    prev_month = 0
    prev_hour = 0
    
    #forecast = np.zeros(12)
    
    # Calculate conditions at each hourly state and store in arrays
    for hour in range(wec.data_length-12):
        state = wec.Hourly_state(hour, ins.solar_panel_specs, ins.wind_turbine_specs)
        
        # Energy flowing to the plant
        energy_generated = wec.calc_generated_kw(state)
        total_renewable += energy_generated
        total_renewable_hourly[state.hour_of_day][state.month-1] += energy_generated
        
        # # Forecast data
        # for j in range(12):
        #     future_state = wec.Hourly_state(hour+j+1, ins.solar_panel_specs, ins.wind_turbine_specs)
        #     forecast[j] = wec.calc_generated_kw(future_state)
        
        # Allow for multiple periods per hour
        for i in range(pph):
            
            # Energy distribution for current period
            energy_tally, r2_e_prev = wec.distribute_energy(energy_generated, 
                                                            energy_tally, 
                                                            r2_e_prev, 
                                                            energy_flow, 
                                                            battery, 
                                                            reactor2,
                                                            ins.r2_max_constants, 
                                                            forecast[hour])
            
            # Update battery charge
            battery.charge += wec.battery_charge_differential(energy_flow.to_battery, battery)
            
            # Calculate reactor 2 state
            r2_sx_current = reactor2.react(energy_flow.to_r2, r2_prev)
            total_sx += r2_sx_current/pph
            r2_prev = r2_sx_current
            total_sx_monthly[state.month-1] += r2_sx_current/pph
            
            # Calculate reactor 1 states and tally changeovers
            r1_changeovers_tally = pc.update_reactor_1(r2_sx_current, r1_changeovers_tally, reactor1_1, reactor1_2)
            
            # Update Sx filter saturation and tally filter changes
            sx_sat, sx_filter_tally = pc.update_sx_filter(sx_sat, sx_filter_tally, sx_filter, r2_sx_current)
            
            # Add up energy taken from grid
            if energy_flow.from_grid > 0:
                e_from_grid += energy_flow.from_grid/pph
                e_from_grid_hourly[state.hour_of_day][state.month-1] += energy_flow.from_grid/pph
                
            if energy_flow.from_grid < 0:
                e_to_grid -= energy_flow.from_grid/pph
                e_to_grid_hourly[state.hour_of_day][state.month-1] -= energy_flow.from_grid/pph
            
            if prev_hour != state.hour_of_day: hour_tally[state.hour_of_day][state.month-1] += 1
            if prev_month != state.month: month_tally[state.month-1] += 1
            
            prev_hour = state.hour_of_day
            prev_month = state.month
    
    # ave_renewable_hourly = total_renewable_hourly / hour_tally
    # ave_from_grid_hourly = e_from_grid_hourly / hour_tally
    # ave_to_grid_hourly = e_to_grid_hourly / hour_tally
    # ave_sx_monthly = total_sx_monthly / month_tally
    
    revenue = 9.6*total_sx + 0.1*e_to_grid
    opex = 0.25*e_from_grid
    capex = 1000*b + 200*sp + (1.5*10**6)*wt
    profit = revenue - opex - capex
    
    return profit, revenue, opex, capex

simulate(wt, sp, b, c1, c2, c3)

num_scenarios = len(wt_list)*len(sp_list)*len(b_list)*len(c1_list)*len(c2_list)*len(c3_list)
results_matrix = np.zeros([10, 10])


end = time.time()
print("Time elapsed: ", end - start)
#28 sec per condition