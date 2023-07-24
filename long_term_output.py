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

# Pre-calculate forecast data
def prep_forecast(parameters):
    wt_list = parameters["wt_list"]
    sp_list = parameters["sp_list"]
    
    forecast_store = np.zeros([len(wt_list), len(sp_list), wec.data_length-6, 2])
    forecast_arr = np.zeros(6)
    
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
    
            for hour in range(wec.data_length-6):
                for k in range(6): 
                    future_state = wec.Hourly_state(hour+k+1, solar_panel_specs, wind_turbine_specs)
                    forecast_arr[k] = wec.calc_generated_kw(future_state)
                
                forecast_store[i][j][hour][0] = sum(forecast_arr[0:3])
                forecast_store[i][j][hour][1] = sum(forecast_arr[3:7])
                
    return forecast_store


# Run DOE scenario to generate profit at specified conditions in 'run'
def run_scenario(forecast_store, parameters, run):
    
    start = time.time()
    
    # factor levels are between -1 and 1, but list indicies are 0-2, so need to
    # add 1 to get index from level. wt is a special case because it's binary
    wt_index = int(run["wt_level"])
    sp_index = int(run["sp_level"]) + 1
    b_index = int(run["b_level"]) + 1
    c1_index = int(run["c1_level"]) + 1
    c2_index = int(run["c2_level"]) + 1
    c3_index = int(run["c3_level"]) + 1
    
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
    
    end = time.time()
    print("Time elapsed: ", end - start)
    
    return profit, revenue, opex, capex, total_sx, e_to_grid, e_from_grid

# This is the CPU intensive function that runs run_scenario function for each 
# run in the DOE
def run_doe(doe, parameters):
    
    doe[["profit", "revenue", "opex", "capex", "total_sx", "e_to_grid", "e_from_grid"]] = np.zeros([len(doe), 7])
    forecast_store = prep_forecast(parameters)
    
    for i in range(len(doe)):
        
        run = doe.iloc[i]
        
        print("Run: ", i)
        
        profit, revenue, opex, capex, total_sx, e_to_grid, e_from_grid = run_scenario(forecast_store, parameters, run)
        doe["profit"][i] = profit
        doe["revenue"][i] = revenue
        doe["opex"][i] = opex
        doe["capex"][i] = capex
        doe["total_sx"][i] = total_sx
        doe["e_to_grid"][i] = e_to_grid
        doe["e_from_grid"][i] = e_from_grid
        
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
#doe.iloc[0]

# Run DOE
doe_results, forecast_store = run_doe(doe, parameters)

#%% Generate a model for profit as a function of the input parameters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.optimize import minimize
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
    sig_factors = est_fit.pvalues[est_fit.pvalues <= 0.05].index
    
    # Ensure all 1st-order terms are present if they're in a higher-level term
    # Start with squared terms
    for factor in sig_factors:
        if "^" in factor:
            first_order_term = re.search("^[a-z0-9]*",factor).group(0)  
            if first_order_term not in sig_factors:
                sig_factors = sig_factors.append(first_order_term)
        
        # Ensure all 1st-order terms from interactions are present
        if "*" in factor:
            first_order_term1 = re.search("^[a-z0-9]*",factor).group(0)
            first_order_term2 = re.search("[a-z0-9]*$",factor).group(0)
            for first_order_term in (first_order_term1, first_order_term2):
                if first_order_term not in sig_factors:
                    sig_factors = sig_factors.append(first_order_term)    
    
    # Create new df of significant factors
    X_sig = X_labeled[sig_factors]
    
    # New model with only significant factors
    est_sig = sm.OLS(y, X_sig)
    est_sig_fit = est_sig.fit()
    
    print("Significant results:")
    print(est_sig_fit.summary())    
    
    return X_sig, est_sig_fit

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


#%% Create Gurobi model to optimize DOE results
model = gp.Model()
model.setParam('NonConvex', 2) # To allow for quadratic equality constraints

factors_dict = {}

# Create gurobi variables
for factor in est_sig_fit.pvalues.index:
    if factor == 'wt':
        factors_dict[factor] = model.addVar(vtype=GRB.BINARY, name=factor)
    elif '^' not in factor and '*' not in factor:
        factors_dict[factor] = model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=factor)
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

test = 0
for var in model.getVars():
    test += var.x * coefs[var.varName]
    print(var.varName, '=', var.x)
print('objective value: ', model.objVal)
print(test)


#%%

# for data in doe, p < 0.05 (significant) are:
# those in the factors_reduced list below
# run again with only these factors + 1st-order terms that appear

X_labeled, y = fit_results(doe_results)

factors_reduced = ["const", "wt", "sp", "b", "c1", "c2", "c3", 
           "wt^2", "wt*sp", "wt*b", "wt*c1", "wt*c2", "wt*c3",
           "sp^2", "sp*b", "b^2",]

X_labeled2 = X_labeled[factors_reduced]

# run again
est2 = sm.OLS(y, X_labeled2)
est_fit2 = est2.fit()
print(est_fit2.summary())

#%%

import sklearn.metrics as metrics 

coefs = est_fit2.params

lr_model2 = LinearRegression().fit(X_labeled2, y)

# objective function is lr_model2.predict(x), but we need to change maximization problem
# to a minimization:
def objective(x): 
    y = coefs[0] + coefs[1]*x[1] + coefs[2]*x[2] + coefs[3]*x[3] + coefs[4]*x[4] \
        + coefs[5]*x[5] + coefs[6]*x[6] + coefs[7]*x[1]**2 + coefs[8]*x[1]*x[2] + coefs[9]*x[1]*x[3] \
        + coefs[10]*x[1]*x[4] + coefs[11]*x[1]*x[5] + coefs[12]*x[1]*x[6] + coefs[13]*x[2]**2 \
        + coefs[14]*x[2]*x[3] + coefs[15]*x[3]**2
    return -y

y_pred = [-objective(X_labeled2.loc[x][:7]) for x in range(len(X_labeled2))]

# test goodness of fit with coef of determination
r_squared = metrics.r2_score(y, y_pred)

# starting guess
x0 = [1,1,1,1,1,1,1]

# constraints

def constraint1(x):
    
    # set wt to 1, as this is binary but it is clear that 1 is better than 0
    min_val = 1
    max_val = 1
    a1 = x[1] - min_val
    a2 = max_val - x[1]
    
    return [a2, a1]

def constraint2(x):
    
    # sp is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[2] - min_val
    a2 = max_val - x[2]
    
    return [a2, a1]

def constraint3(x):
    
    # b is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[3] - min_val
    a2 = max_val - x[3]
    
    return [a2, a1]

def constraint4(x):
    
    # c1 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[4] - min_val
    a2 = max_val - x[4]
    
    return [a2, a1]

def constraint5(x):
    
    # c2 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[5] - min_val
    a2 = max_val - x[5]
    
    return [a2, a1]

def constraint6(x):
    
    # c3 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[6] - min_val
    a2 = max_val - x[6]
    
    return [a2, a1]
    
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3},
               {'type': 'ineq', 'fun': constraint4},
               {'type': 'ineq', 'fun': constraint5},
               {'type': 'ineq', 'fun': constraint6}
              ]

res = minimize(objective, x0, method='COBYLA', constraints = constraints)

parameters = {
    "wt_list" : [0, 1], # number of 1MW wind turbines
    "sp_list" : [5000, 10000, 15000], # area in m2 of solar panels
    "b_list" : [516, 1144, 2288], # battery sizes in kW
    "c1_list" : [0, 1, 2], # constants for r2_max eqn
    "c2_list" : [0, 1, 2],
    "c3_list" : [-1, 0, 1]
    }

res_x = res.x
res_y = -objective(res_x)


run = pd.Series([1, 0, 0, 2, 0, 2], ["wt_level", "sp_level", "b_level", "c1_level",
                                     "c2_level", "c3_level"])
pred_profit, pred_revenue, pred_opex, pred_capex, pred_total_sx, pred_e_to_grid, \
    pred_e_from_grid = run_scenario(forecast_store, parameters, run)

#%% 2nd DOE
# Since the optimal solution is at one corner of our DOE, I'm updating the 
# parameter values to see if we can get a more optimal solution

parameters2 = {
    "wt_list" : [1, 2], # number of 1MW wind turbines
    "sp_list" : [2000, 5000, 8000], # area in m2 of solar panels
    "b_list" : [263, 516, 1144], # battery sizes in kW
    "c1_list" : [1, 2, 3], # constants for r2_max eqn
    "c2_list" : [-1, 0, 1],
    "c3_list" : [0, 1, 2]
    }

# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe2 = pd.read_excel("DOE.xlsx")
#doe.iloc[0]

# Run DOE
doe_results2, forecast_store = run_doe(doe2, parameters2)

#%% Run with exactly 1 wind turbine
# A 2nd wind turbine increases revenue by delivering a large amount of energy to
# the grid. However, the goal of this plant is Sx production, so we're going to 
# limit to 1 wind turbine and re-run the DOE, from DOE2.xlsx

parameters3 = {
    "wt_list" : [1, 1], # number of 1MW wind turbines
    "sp_list" : [2000, 5000, 8000], # area in m2 of solar panels
    "b_list" : [263, 516, 1144], # battery sizes in kW
    "c1_list" : [1, 2, 3], # constants for r2_max eqn
    "c2_list" : [-1, 0, 1],
    "c3_list" : [0, 1, 2]
    }

# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe2 = pd.read_excel("DOE2.xlsx")
#doe.iloc[0]

# Run DOE
doe_results2, forecast_store = run_doe(doe2, parameters3)

X_labeled, y = fit_results(doe_results2)

#%%
factors_reduced2 = ["const", "sp", "b", "c2", "sp^2", "sp*b", "sp*c2", "b^2", "b*c2"]

X_labeled2_2 = X_labeled[factors_reduced2]

# run again
est2 = sm.OLS(y, X_labeled2_2)
est_fit2 = est2.fit()
print(est_fit2.summary())

#%%
coefs = est_fit2.params

lr_model2 = LinearRegression().fit(X_labeled2_2, y)

# objective function is lr_model2.predict(x), but we need to change maximization problem
# to a minimization:
def objective2(x): 
    y = coefs[0] + coefs[1]*x[1] + coefs[2]*x[2] + coefs[3]*x[3] + coefs[4]*x[1]**2 \
        + coefs[5]*x[1]*x[2] + coefs[6]*x[1]*x[3] + coefs[7]*x[2]**2 + coefs[8]*x[2]*x[3]
    return -y

y_pred = [-objective2(X_labeled2_2.loc[x][:4]) for x in range(len(X_labeled2_2))]

# test goodness of fit with coef of determination
r_squared = metrics.r2_score(y, y_pred)

#%% optimize # 2

# starting guess
x0 = [1,1,1,1]

# constraints

def constraint1_2(x):
    
    # sp is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[1] - min_val
    a2 = max_val - x[1]
    
    return [a2, a1]

def constraint2_2(x):
    
    # b is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[2] - min_val
    a2 = max_val - x[2]
    
    return [a2, a1]

def constraint3_2(x):
    
    # c2 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[3] - min_val
    a2 = max_val - x[3]
    
    return [a2, a1]

    
constraints = [{'type': 'ineq', 'fun': constraint1_2},
               {'type': 'ineq', 'fun': constraint2_2},
               {'type': 'ineq', 'fun': constraint3_2},
              ]

res = minimize(objective2, x0, method='COBYLA', constraints = constraints)

res_x = res.x
res_y = -objective2(res_x)

'''
Results of this new DOE are sp = 1.38, b = 0, and c2 = 1. 
Since we're still on the edge for the battery, run again with a lower range for
the battery. Will also shift sp to center around 1.38
'''
#%% Shifting range of sp and b and rerunning
parameters3 = {
    "wt_list" : [1, 1], # number of 1MW wind turbines
    "sp_list" : [4000, 6500, 9000], # area in m2 of solar panels
    "b_list" : [0, 263, 516], # battery sizes in kW
    "c1_list" : [1, 2, 3], # constants for r2_max eqn
    "c2_list" : [-1, 0, 1],
    "c3_list" : [0, 1, 2]
    }

# DOE input is a 2-level full factorial plus a Box-Behnken to capture curvature
doe2 = pd.read_excel("DOE2.xlsx")
#doe.iloc[0]

# Run DOE
doe_results3, forecast_store = run_doe(doe2, parameters3)

X_labeled, y = fit_results(doe_results3)

#%%
factors_reduced3 = ["const", "sp", "b", "c2", "sp^2", "sp*b", "sp*c2", "b^2", "b*c2"]

X_labeled2_3 = X_labeled[factors_reduced3]

# run again
est2 = sm.OLS(y, X_labeled2_3)
est_fit2 = est2.fit()
print(est_fit2.summary())

#%%
# can reduce 2 more factors
factors_reduced3 = ["const", "sp", "b", "c2", "sp^2", "sp*b", "sp*c2"]

X_labeled2_3 = X_labeled[factors_reduced3]

# run again
est2 = sm.OLS(y, X_labeled2_3)
est_fit2 = est2.fit()
print(est_fit2.summary())

#%%
coefs = est_fit2.params

lr_model2 = LinearRegression().fit(X_labeled2_3, y)

# objective function is lr_model2.predict(x), but we need to change maximization problem
# to a minimization:
def objective3(x): 
    y = coefs[0] + coefs[1]*x[1] + coefs[2]*x[2] + coefs[3]*x[3] + coefs[4]*x[1]**2 \
        + coefs[5]*x[1]*x[2] + coefs[6]*x[1]*x[3]
    return -y

y_pred = [-objective3(X_labeled2_2.loc[x][:4]) for x in range(len(X_labeled2_2))]

# test goodness of fit with coef of determination
r_squared = metrics.r2_score(y, y_pred)

#%% optimize # 2

# starting guess
x0 = [1,1,1,1]

# constraints

def constraint1_2(x):
    
    # sp is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[1] - min_val
    a2 = max_val - x[1]
    
    return [a2, a1]

def constraint2_2(x):
    
    # b is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[2] - min_val
    a2 = max_val - x[2]
    
    return [a2, a1]

def constraint3_2(x):
    
    # c2 is between 0 and 2
    min_val = 0
    max_val = 2
    a1 = x[3] - min_val
    a2 = max_val - x[3]
    
    return [a2, a1]

    
constraints = [{'type': 'ineq', 'fun': constraint1_2},
               {'type': 'ineq', 'fun': constraint2_2},
               {'type': 'ineq', 'fun': constraint3_2},
              ]

res = minimize(objective3, x0, method='COBYLA', constraints = constraints)

res_x = res.x
res_y = -objective3(res_x)


'''
according to this, sp is centered (1.04), and battery and c2 are at 0. Lower
limit for battery was already 0, so we can't go any lower
'''
#%% test these parameters on the training data
run = pd.Series([1, 1, 0, 1, 0, 1], ["wt_level", "sp_level", "b_level", "c1_level",
                                     "c2_level", "c3_level"])
opt_profit, opt_revenue, opt_opex, opt_capex, opt_total_sx, opt_e_to_grid, opt_e_from_grid \
    = run_scenario(forecast_store, parameters3, run)
    
#%% New attempt
    