# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:47:37 2023

@author: rhanusa
"""

import numpy as np
import plant_components as pc
from plant_components import pph
import weather_energy_components as wec
import time
import input_specs as ins
import pandas as pd

# Pre-calculate forecast data
def prep_forecast(parameters):
    wt_list = parameters["wt_list"]
    sp_list = parameters["sp_list"]
    
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
                
    return forecast_store


# Run DOE scenario to generate profit at specified conditions in 'run'
def run_scenario(forecast_store, parameters, run):
    
    start = time.time()
    
    wt_level = int(run["wt_level"])
    sp_level = int(run["sp_level"])
    b_level = int(run["b_level"])
    c1_level = int(run["c1_level"])
    c2_level = int(run["c2_level"])
    c3_level = int(run["c3_level"])
    
    wt_list = parameters["wt_list"]
    sp_list = parameters["sp_list"]
    b_list = parameters["b_list"]
    c1_list = parameters["c1_list"]
    c2_list = parameters["c2_list"]
    c3_list = parameters["c3_list"]
    
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
    
    # Initiate necessary variables
    r2_prev = 0 
    reactor1_1 = pc.Reactor1()
    reactor1_1.state = "active"
    reactor2 = pc.Reactor2()
    battery = pc.Battery(0.5*battery_specs["max_charge"], battery_specs)
    energy_flow = wec.Energy_flow()
    energy_tally = pph
    r2_e_prev = 0
    
    # Calculate conditions at each hourly state and store in arrays
    for hour in range(wec.data_length-12):
        state = wec.Hourly_state(hour, ins.solar_panel_specs, ins.wind_turbine_specs)
        
        # Energy flowing to the plant
        energy_generated = wec.calc_generated_kw(state)
        total_renewable += energy_generated
        total_renewable_hourly[state.hour_of_day][state.month-1] += energy_generated
        
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
            
            # Add up energy taken from grid
            if energy_flow.from_grid > 0:
                e_from_grid += energy_flow.from_grid/pph
                e_from_grid_hourly[state.hour_of_day][state.month-1] += energy_flow.from_grid/pph
                
            if energy_flow.from_grid < 0:
                e_to_grid -= energy_flow.from_grid/pph
                e_to_grid_hourly[state.hour_of_day][state.month-1] -= energy_flow.from_grid/pph
    
    # spread capex cost over 10 years, so cost per year is always /10
    capex = (battery_specs["cost"] + solar_panel_specs["cost"] + wind_turbine_specs["cost"])/10
    revenue = (9.6*total_sx + 0.1*e_to_grid)/8 # divide by 8 years because 8 years of training data, -> revenue / year
    
    # Roughly 15 kW required to make 1 mol S, so changing coef to .75 instead of .25
    # will make it a little unfavorable to buy from grid to make Sx.
    # With .75 coef there was still a favor of using grid energy to make Sx, but 
    # Somewhat less so, since the battery was large in the optimal configureation
    # 3rd attempt now with higher penalty at 1.25
    opex = 1.25*e_from_grid/8 # divide by 8 years because 8 years of training data, -> opex / year
    
    profit = revenue - opex - capex
    
    end = time.time()
    print("Time elapsed: ", end - start)
    
    return profit, revenue, opex, capex

# This is the CPU intensive function that runs run_scenario function for each 
# run in the DOE
def run_doe(doe, parameters):
    
    doe[["profit", "revenue", "opex", "capex"]] = np.zeros([len(doe), 4])
    forecast_store = prep_forecast(parameters)
    
    for i in range(len(doe)):
        
        run = doe.iloc[i]
        
        print("Run: ", i)
        
        profit, revenue, opex, capex = run_scenario(forecast_store, parameters, run)
        doe["profit"][i] = profit
        doe["revenue"][i] = revenue
        doe["opex"][i] = opex
        doe["capex"][i] = capex
        
    return doe


# parameters for doe_results_202230706.csv
parameters = {
    "wt_list" : [0, 1], # number of 1MW wind turbines
    "sp_list" : [5000, 10000, 15000], # area in m2 of solar panels
    "b_list" : [516, 1144, 2288], # battery sizes in kW
    "c1_list" : [-.05, 0.025, 0.1], # constants for r2_max eqn
    "c2_list" : [-.05, 0.025, 0.1],
    "c3_list" : [-2*10**(-5), 1*10**(-5), 4*10**(-5)]
    }

# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe = pd.read_excel("DOE.xlsx")
doe.iloc[0]

# Run DOE
doe_results = run_doe(doe, parameters)
    
#%% Generate a model for profit as a function of the input parameters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = doe_results[["wt_level", "sp_level", "b_level", "c1_level", "c2_level", "c3_level"]]
y = doe_results["profit"]

model = polyfeatures3 = PolynomialFeatures(degree=2)
fit_tr_model = model.fit_transform(X)
lr_model = LinearRegression()
lr_model.fit(fit_tr_model, y)

# These results show that the most profitable is highest levels on sp, wt, and
# All c constants, and lowest level on battery. Basically, it was most profitable
# to purchase a lot of grid energy to produce Sx
# Next, I'll try to allow negative c values, and do a higher penalty for grid energy
# Also, changing to ave profit / year, and spread cost of capex out over 10 years.

# Re-ran with higher penalty for grid energy. Now the maximum profit is at the 
# max for all capex and all c's. The grid energy expenditure is higher than most,
# but not astronomical like the last version (likely because we use a large battery
# here)

#%% Optimization: find parameter values that maximize profit
import statsmodels.api as sm
from scipy.optimize import minimize

factors = ["const", "wt", "sp", "b", "c1", "c2", "c3", 
           "wt^2", "wt*sp", "wt*b", "wt*c1", "wt*c2", "wt*c3",
           "sp^2", "sp*b", "sp*c1", "sp*c2", "sp*c3",
           "b^2", "b*c1","b*c2","b*c3",
           "c1^2", "c1*c2", "c1*c3", "c2^2", "c2*c3", "c3^2"]

X_labeled = pd.DataFrame(fit_tr_model, columns=factors)

est = sm.OLS(y, X_labeled)
est_fit = est.fit()
print(est_fit.summary())

# for data in doe = doe_results_20230706, p < 0.10 (significant) are:
# wt, wt^2, wt*c1, wt*c2, wt*c3, b*c3
# run again with only these factors + 1st-order terms that appear

factors_reduced = ["const", "wt", "b", "c1", "c2", "c3", 
                   "wt^2", "wt*c1", "wt*c2", "wt*c3", "b*c3"]

X_labeled2 = X_labeled[factors_reduced]

# run again
est2 = sm.OLS(y, X_labeled2)
est_fit2 = est2.fit()
print(est_fit2.summary())

# All terms have p < 0.10 except for c3, but we leave it in because it's part of interaction terms
coefs = est_fit2.params

lr_model2 = LinearRegression().fit(X_labeled2, y)

# objective function is lr_model2.predict(x), but we need to change maximization problem
# to a minimization:
def objective(x):
    y = coefs[0] + coefs[1]*x[1] + coefs[2]*x[2] + coefs[3]*x[3] + coefs[4]*x[4] \
        + coefs[5]*x[5] + coefs[6]*x[1]**2 + coefs[7]*x[1]*x[3] + coefs[8]*x[1]*x[4] \
        + coefs[9]*x[1]*x[5] + coefs[10]*x[2]*x[5]
    return -y

test = [objective(X_labeled2.loc[x][:6]) for x in range(len(X_labeled2))]

# starting guess
x0 = [1,1,1,1,1,1]

# constraints

def constraint1(x):
    
    # set wt to 1, as this is binary but it is clear that 1 is better than 0
    min_val = 1
    max_val = 1
    a1 = x[1] - min_val
    a2 = max_val - x[1]
    
    return [a2, a1]

def constraint2(x):
    
    # b is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[2] - min_val
    a2 = max_val - x[2]
    
    return [a2, a1]

def constraint3(x):
    
    # c1 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[3] - min_val
    a2 = max_val - x[3]
    
    return [a2, a1]

def constraint4(x):
    
    # c2 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[4] - min_val
    a2 = max_val - x[4]
    
    return [a2, a1]

def constraint5(x):
    
    # c3 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[5] - min_val
    a2 = max_val - x[5]
    
    return [a2, a1]
    
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3},
               {'type': 'ineq', 'fun': constraint4},
               {'type': 'ineq', 'fun': constraint5},
              ]

res = minimize(objective, x0, method='COBYLA', constraints = constraints)

# Result is (1, 1, 2, 2, 2, 2) for ["const", "wt", "b", "c1", "c2", "c3"]

res_x = res.x
res_y = -objective(res_x)
test_y = run_scenario(1, 1, 2, 2, 2, 2)

