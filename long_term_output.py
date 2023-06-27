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


# Inputs
battery_specs = { # Using battery at https://www.backupbatterypower.com/products/1-144-kwh-industrial-battery-backup-and-energy-storage-systems-ess-277-480-three-phase?pr_prod_strat=use_description&pr_rec_id=97fba9aea&pr_rec_pid=6561192214696&pr_ref_pid=6561191100584&pr_seq=uniform
    "max_charge": 1144, # kWh
    "cost": 1.2*10**6 
    }

solar_panel_specs = {
    "area": 10000, # m^2
    "efficiency": 0.1,
    "cost": 10000*200/1.1 # $200/m2 (/1.1 eur to usd) https://www.sunpal-solar.com/info/how-much-does-a-solar-panel-cost-per-square-me-72064318.html
    }

wind_turbine_specs = {
    "cut_in": 13, # km/h
    "rated_speed": 50, # km/h
    "cut_out": 100, # km/h
    "max_energy": 1000, # kW
    "count": 1,
    "cost": 1.5*10**6 # EUR/MW (https://www.windustry.org/how_much_do_wind_turbines_cost)
    }

# initialize counters
total_grid = 0 # kw
total_grid_hourly = np.zeros([24,12]) # kw per hour per month
total_renewable = 0 # kw
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
battery = pc.Battery(0.5*wec.battery_max, wec.battery_max)
energy_flow = wec.Energy_flow()
energy_tally = pph
r2_e_prev = 0
prev_month = 0
prev_hour = 0

# Calculate conditions at each hourly state and store in arrays
for hour in range(wec.data_length):
    state = wec.Hourly_state(hour)
    
    # Allow for multiple periods per hour
    for i in range(pph):
        
        # Energy flowing to the plant
        energy_generated = wec.calc_generated_kw(state)
        total_renewable += energy_generated/pph
        total_renewable_hourly[state.hour_of_day][state.month-1] += energy_generated/pph
        
        # Energy distribution for current period
        energy_tally, r2_e_prev = wec.distribute_energy(energy_generated, energy_tally, r2_e_prev, energy_flow, battery, reactor2)
        
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
        total_grid += energy_flow.from_grid/pph
        total_grid_hourly[state.hour_of_day][state.month-1] += energy_flow.from_grid/pph
        
        if prev_hour != state.hour_of_day: hour_tally[state.hour_of_day][state.month-1] += 1
        if prev_month != state.month: month_tally[state.month-1] += 1
        
        prev_hour = state.hour_of_day
        prev_month = state.month

ave_renewable_hourly = total_renewable_hourly / hour_tally
ave_grid_hourly = total_grid_hourly / hour_tally
ave_sx_monthly = total_sx_monthly / month_tally

#%% Plot grid consumptions

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default='browser'

months = ['January',
          'February',
          'March',
          'April',
          'May',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December']

#colors for legend
colors = ['#1e00ff',
        '#a300e5',
        '#dd00c6',
        '#ff00a5',
        '#ff0085',
        '#ff0067',
        '#ff004c',
        '#ff4b31',
        '#ff7a0d',
        '#ff9e00',
        '#ffbd00',
        '#ffd800']

fig = make_subplots(rows=1,cols=1)
for i in range(len(months)):
    fig.add_trace(go.Scatter(x=[x for x in range(24)], y=ave_grid_hourly[:,i], 
                             mode="lines", name=months[i], line_color=colors[i]),
                  row=1, col=1)
    
fig.update_xaxes(title_text="Time of day (hour)",row=1, col=1)
fig.update_yaxes(title_text="Average kWh from grid", range=[-1,21],row=1, col=1)
fig.update_layout(title_text="Average energy needed from grid per hour by month", title_x=0.5)

fig.show()
