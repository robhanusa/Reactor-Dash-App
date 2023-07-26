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
import pandas as pd

# Number of years the model spans
years = 8

# Capex lifetime to calculate value of capex per year
capex_life = 10

def generate_wt_specs(wt):
    wind_turbine_specs = {
        "cut_in": 13, # km/h
        "rated_speed": 50, # km/h
        "cut_out": 100, # km/h
        "max_energy": 1000, # kW
        "count": wt,
        "cost": wt*1.5*10**6 # EUR/MW 
        }
    
    return wind_turbine_specs

def generate_sp_specs(sp):
    solar_panel_specs = {
        "area": sp, # m^2
        "efficiency": 0.1,
        "cost": sp*200/1.1 # $200/m2 (/1.1 eur to usd)
        }
    
    return solar_panel_specs

def generate_b_specs(b):
    battery_specs = { 
        "max_charge": b, # kWh
        "cost": 1000*b
        }
    
    return battery_specs

# Pre-calculate forecast data
def prep_forecast(parameters):
    wt_list = parameters["wt_list"]
    sp_list = parameters["sp_list"]
    
    forecast_store = np.zeros([len(wt_list), len(sp_list), wec.data_length-6, 2])
    forecast_arr = np.zeros(6)
    
    for i in range(len(wt_list)):
        wt = wt_list[i]
        wind_turbine_specs = generate_wt_specs(wt)
        
        for j in range(len(sp_list)):
            sp = sp_list[j]
            solar_panel_specs = generate_sp_specs(sp)
    
            for hour in range(wec.data_length-6):
                for k in range(6): 
                    future_state = wec.Hourly_state(hour+k+1, solar_panel_specs, wind_turbine_specs)
                    forecast_arr[k] = wec.calc_generated_kw(future_state)
                
                forecast_store[i][j][hour][0] = sum(forecast_arr[0:3])
                forecast_store[i][j][hour][1] = sum(forecast_arr[3:7])
                
    return forecast_store


# Run DOE scenario to generate profit at specified conditions in 'run'
def run_scenario(forecast_store, parameters, run):
    
    # factor levels are between 0 and 2 to match indicies 
    # wt is a special case because only 2 levels are considered
    wt_index = int(run["wt_level"]) if int(run["wt_level"]) == 0 else 1
    sp_index = int(run["sp_level"]) 
    b_index = int(run["b_level"])
    c1_index = int(run["c1_level"])
    c2_index = int(run["c2_level"])
    c3_index = int(run["c3_level"])
    
    wt_list = parameters["wt_list"]
    sp_list = parameters["sp_list"]
    b_list = parameters["b_list"]
    c1_list = parameters["c1_list"]
    c2_list = parameters["c2_list"]
    c3_list = parameters["c3_list"]
    
    wt = wt_list[wt_index]
    sp = sp_list[sp_index]
    b = b_list[b_index]
    c1 = c1_list[c1_index]
    c2 = c2_list[c2_index]
    c3 = c3_list[c3_index]
    
    battery_specs = generate_b_specs(b)

    solar_panel_specs = generate_sp_specs(sp)
    
    wind_turbine_specs = generate_wt_specs(wt)
    
    b_sp_constants = {
        "c1": c1,
        "c2": c2,
        "c3": c3
        }

    # initialize counters
    e_from_grid = 0 # kwh
    e_to_grid = 0 # kwh
    total_renewable = 0 # kwh
    total_sx = 0 # mol
    
    # Initiate necessary variables
    r2_prev = 0 
    reactor1_1 = pc.Reactor1()
    reactor1_1.state = "active"
    reactor2 = pc.Reactor2()
    battery = pc.Battery(0.5*battery_specs["max_charge"], battery_specs)
    energy_flow = wec.Energy_flow()
    energy_tally = pph
    r2_e_prev = 0
    p_renew_tmin1 = 0
    
    # Calculate conditions at each hourly state and store in arrays
    for hour in range(wec.data_length-12):
        state = wec.Hourly_state(hour, solar_panel_specs, wind_turbine_specs)
        
        # Energy flowing to the plant
        p_renew_t_actual = wec.calc_generated_kw(state)
        total_renewable += p_renew_t_actual
        
        forecast = (forecast_store[wt_index][sp_index][hour][0],
                    forecast_store[wt_index][sp_index][hour][1])
        
        # Allow for multiple periods per hour
        for i in range(pph):
            
            # Energy distribution for current period
            energy_tally, r2_e_prev, energy_flow = wec.distribute_energy(p_renew_t_actual,
                                                            p_renew_tmin1,
                                                            energy_tally, 
                                                            r2_e_prev, 
                                                            energy_flow, 
                                                            battery, 
                                                            b_sp_constants,
                                                            reactor2,
                                                            forecast)
                   
            # Update battery charge
            battery.charge += wec.battery_charge_differential(energy_flow.to_battery, battery)
            
            # Calculate reactor 2 state
            r2_sx_current = reactor2.react(energy_flow.to_r2, r2_prev)
            total_sx += r2_sx_current/pph
            
            r2_prev = r2_sx_current
            
            # Add up energy taken from grid
            if energy_flow.from_grid > 0:
                e_from_grid += energy_flow.from_grid/pph
                
            if energy_flow.from_grid < 0:
                e_to_grid -= energy_flow.from_grid/pph
                
        p_renew_tmin1 = p_renew_t_actual
    
    # spread capex cost over 10 years, so cost per year is always /10
    capex = (battery_specs["cost"] + solar_panel_specs["cost"] + wind_turbine_specs["cost"])/capex_life
    
    # Target production is 240 kmol S per year (1.92 million for 8 years). So, assume that S above this 
    # value is worth 0. Otherwise, it always becomes advantageous to produce more
    # S, even if we just use grid energy to do it.
    revenue = (9.6*min(total_sx, 240000*years) + 0.1*e_to_grid)/years # divide by 8 years because 8 years of training data, -> revenue / year
    
    # Roughly 15 kW required to make 1 mol S
    opex = 0.25*e_from_grid/years # divide by 8 years because 8 years of training data, -> opex / year
    
    profit = revenue - opex - capex
    
    return profit, revenue, opex, capex, total_sx, e_to_grid, e_from_grid

# This is the CPU intensive function that runs run_scenario function for each 
# run in the DOE
def run_doe(doe, parameters, forecast_store = 0, show_run_status = True ):
    
    doe[["profit", "revenue", "opex", "capex", "total_sx", "e_to_grid", "e_from_grid"]] = np.zeros([len(doe), 7])
    
    if not isinstance(forecast_store, np.ndarray):
        forecast_store = prep_forecast(parameters)
    
    for i in range(len(doe)):
        
        start = time.time()
        
        run = doe.iloc[i]
        
        profit, revenue, opex, capex, total_sx, e_to_grid, e_from_grid = run_scenario(forecast_store, parameters, run)
        doe["profit"][i] = profit
        doe["revenue"][i] = revenue
        doe["opex"][i] = opex
        doe["capex"][i] = capex
        doe["total_sx"][i] = total_sx
        doe["e_to_grid"][i] = e_to_grid
        doe["e_from_grid"][i] = e_from_grid
        
        end = time.time()
        
        if show_run_status: 
            print("Run: ", i)
            print("Time elapsed: ", end - start)
        
    return doe, forecast_store

parameters = {
    "wt_list" : [0, 1], # number of 1MW wind turbines
    "sp_list" : [5000, 10000, 15000], # area in m2 of solar panels
    "b_list" : [516, 1144, 2288], # battery sizes in kW
    "c1_list" : [0, 1, 2], # constants for r2_max eqn
    "c2_list" : [0, 1, 2],
    "c3_list" : [-1, 0, 1]
    }

# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe = pd.read_excel("DOE.xlsx")

# Run DOE
doe_results, forecast_store = run_doe(doe, parameters) #, forecast_store)

#%% Generate a model for profit as a function of the input parameters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import re
import gurobipy as gp
from gurobipy import GRB

def fit_results(doe_results):
    X = doe_results[["wt_level", "sp_level", "b_level", "c1_level", "c2_level", "c3_level"]]
    y = doe_results["profit"]
    
    model = PolynomialFeatures(degree=2)
    fit_tr_model = model.fit_transform(X)
    lr_model = LinearRegression()
    lr_model.fit(fit_tr_model, y)
    
    factors = ["const", "wt", "sp", "b", "c1", "c2", "c3", 
               "wt^2", "wt*sp", "wt*b", "wt*c1", "wt*c2", "wt*c3",
               "sp^2", "sp*b", "sp*c1", "sp*c2", "sp*c3",
               "b^2", "b*c1","b*c2","b*c3",
               "c1^2", "c1*c2", "c1*c3", "c2^2", "c2*c3", "c3^2"]
    
    X_labeled = pd.DataFrame(fit_tr_model, columns=factors)
    
    est = sm.OLS(y, X_labeled)
    est_fit = est.fit()
    print("Initial results:")
    print(est_fit.summary())
    return X_labeled, y, est_fit

def fit_sig_results(X_labeled, y, est_fit):
    
    # filter non-significant factors
    sig_factors = list(est_fit.pvalues[est_fit.pvalues <= 0.05].index)
    
    # Ensure all 1st-order terms are present if they're in a higher-level term
    # Start with squared terms
    for factor in sig_factors:
        if "^" in factor:
            first_order_term = re.search("^[a-z0-9]*",factor).group(0)  
            if first_order_term not in sig_factors:
                sig_factors.append(first_order_term)
        
        # Ensure all 1st-order terms from interactions are present
        if "*" in factor:
            first_order_term1 = re.search("^[a-z0-9]*",factor).group(0)
            first_order_term2 = re.search("[a-z0-9]*$",factor).group(0)
            for first_order_term in (first_order_term1, first_order_term2):
                if first_order_term not in sig_factors:
                    sig_factors.append(first_order_term)    
    
    # Create new df of significant factors
    X_sig = X_labeled[sig_factors]
    
    # New model with only significant factors
    est_sig = sm.OLS(y, X_sig)
    est_sig_fit = est_sig.fit()
    
    print("Significant results:")
    print(est_sig_fit.summary())    
    
    return X_sig, est_sig_fit

# Generate a pd.Series of coefficients for only the significant factors and 
# their 1st-order terms
def generate_sig_model(doe_results):
    
    # Fit results on all factors
    X_labeled, y, est_fit = fit_results(doe_results)  
    
    est_fit_old = est_fit
    
    # Re-fit results on only significant factors (and their first order components)
    X_sig, est_sig_fit = fit_sig_results(X_labeled, y, est_fit)
    
    # Keep eliminating factors until there is no change in the set of factors
    break_counter = 0
    while len(est_fit_old.pvalues) != len(est_sig_fit.pvalues) and break_counter < 10:
        break_counter += 1
        est_fit_old = est_sig_fit
        X_sig, est_sig_fit = fit_sig_results(X_sig, y, est_sig_fit)
    
    coefs = est_sig_fit.params
    
    return coefs

coefs = generate_sig_model(doe_results)

#%% Create Gurobi model to optimize DOE results

# Create Gurobi environment and suppress output
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

# Optimizes model based on the series of coefficients for each factor.
# Returns a Gurobi Model object and prints the optimal factor values
def optimize_model(coefs):
    model = gp.Model(env=env)
    model.setParam('NonConvex', 2) # To allow for quadratic equality constraints
    
    factors_dict = {}
    
    # Create gurobi variables
    for factor in coefs.index:
        if factor == 'wt':
            factors_dict[factor] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=factor)
        elif '^' not in factor and '*' not in factor:
            factors_dict[factor] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=factor)
        else:
            factors_dict[factor] = model.addVar(vtype=GRB.CONTINUOUS, name=factor)
    
    # Add equality constraints
    for f in factors_dict:
        if f == 'const': # Must maintain constant term, constrain this to 1
            factor = factors_dict[f]
            model.addConstr(factor == 1)
        if '^' in f: # Squared terms must be equal to the square of the first-order term
            first_order_f = re.search("^[a-z0-9]*", str(f)).group(0) 
            first_order_factor = factors_dict[first_order_f]
            factor = factors_dict[f]
            model.addConstr(factor == first_order_factor**2)
        if '*' in f: # Interaction terms must be the product of the first-order terms
            first_order_f1 = re.search("^[a-z0-9]*", str(f)).group(0)
            first_order_f2 = re.search("[a-z0-9]*$", str(f)).group(0)
            first_order_factor1 = factors_dict[first_order_f1]
            first_order_factor2 = factors_dict[first_order_f2]
            factor = factors_dict[f]
            model.addConstr(factor == first_order_factor1 * first_order_factor2)
            
    # Objective function
    model.setObjective(gp.quicksum(coefs[factor]*factors_dict[factor] for factor in factors_dict),
                        GRB.MAXIMIZE)    
    
    model.optimize()
    
    for var in model.getVars():
        print(var.varName, '=', var.x)
    print('objective value: ', model.objVal)
    
    return model

model = optimize_model(coefs)

#%% Optimal model is at corner point, so re-run with new parameter values, 
# setting wind turbine equal to 1. The results of the first DOE
# make it clear that 1 wind turbine is better than 0. Because they're so expensive,
# we'll assume that a 2nd isn't an option. In practice, a 2nd wind turbine might
# increase revenue by producing a large excess of energy for the grid. As the goal
# of this plant is to produce Sx and not energy, it is reasonable to limit the 
# model to 1 wind turbine
# Because of this, we can also reduce the size of the DOE to eliminate wt = 0.
# This is saved in DOE2.xlsx

parameters2 = {
    "wt_list" : [1, 1], # number of 1MW wind turbines
    "sp_list" : [2000, 5000, 8000], # area in m2 of solar panels
    "b_list" : [263, 516, 1144], # battery sizes in kW
    "c1_list" : [1, 2, 3], # constants for battery setpoint eqn
    "c2_list" : [-1, 0, 1],
    "c3_list" : [0, 1, 2]
    }

# New DOE which removes the wind turbine factor (i.e. sets it always equal to 1)
doe2 = pd.read_excel("DOE2.xlsx")

# Run DOE
doe_results2, _ = run_doe(doe2, parameters2, forecast_store)

coefs2 = generate_sig_model(doe_results2)

model2 = optimize_model(coefs2)

#%%

parameters3 = {
    "wt_list" : [1, 1], # Only the case with 1 wind turbine is considered
    "sp_list" : [4000, 6500, 9000], # area in m2 of solar panels
    "b_list" : [0, 263, 516], # battery sizes in kW
    "c1_list" : [1, 2, 3], # constants for battery setpoint eqn
    "c2_list" : [-1, 0, 1],
    "c3_list" : [0, 1, 2]
    }

# DOE form doesn't change, so no new DOE is loaded

# Run DOE
doe_results3, _ = run_doe(doe2, parameters3, forecast_store)

coefs3 = generate_sig_model(doe_results3)

model3 = optimize_model(coefs3)

#%% Results suggest it is optimal to have no battery. This is reasonable, as it
# suggests the cost of the battery is too high to be offset by the energy from 
# grid we must buy when renewable energy production is low.
# As all the constants relate to parameters for determining battery set point, 
# they are no longer relevant in the model and will be set to 0 for the next 
# calculation.
# Below, we calculate the expected results at this optimal level

parameters_final = {
    "wt_list" : [1], # Only the case with 1 wind turbine is considered
    "sp_list" : [6500], # area in m2 of solar panels
    "b_list" : [0], # battery sizes in kW
    "c1_list" : [0], # constants for battery setpoint eqn
    "c2_list" : [0],
    "c3_list" : [0]
    }

run = pd.Series([0, 0, 0, 0, 0, 0], ["wt_level", "sp_level", "b_level", "c1_level",
                                     "c2_level", "c3_level"])

profit, revenue, opex, capex, total_sx, e_to_grid, e_from_grid \
    = run_scenario(forecast_store, parameters_final, run)
    
print("Profit (€/yr): ", round(profit))
print("Revenue (€/yr): ", round(revenue))
print("Opex (€/yr): ", round(opex))
print("Capex (€/yr): ", round(capex))
print("Sulfur (kmol/yr): ", round(total_sx/years/1000))
print("Energy sold to grid (MW/yr): ", round(e_to_grid/years/1000, 1))
print("Energy purchased from grid (MW/yr): ", round(e_from_grid/years/1000, 1))