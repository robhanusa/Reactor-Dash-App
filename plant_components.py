# -*- coding: utf-8 -*-
"""
Created on Wed May 17 07:44:15 2023

@author: rhanusa
"""
import math

# periods per hour. Reactor states will be calculated "pph" times per hour. eg
# if pph = 10, states are calculated every 1hr/10 = 6 min. This is necessary because 
# the weather data hourly, but plant state is calculated more frequently
pph = 4

r1_cleaning_speed = 0.1

class Battery:
    def __init__(self, charge, max_charge):
        self.charge = charge
        self.max_charge = max_charge
        
        # Note that battery efficiency (below) is normally dependent on charge. 
        # For a more accurate model, I should account for this
        self.efficiency = 0.9 
    
class Reactor1():
    def __init__(self):
        self.state = "idle"
        self.saturation = 0
        self.sat_factor = 5 # How fast r1 gets saturated per mol COS produced
        self.clean_speed = 10 # How fast r1 gets cleaned
        
    @classmethod
    
    #need to do something about the ss graph for r1. No longer logical
    def ss_output(cls, energy):
        return 2/(1 + math.exp(-energy + 1))-0.54
    
    def add_water(cos_produced):
        Reactor1.state = "active"
    
    def react(cls, sx_produced):
        cos_produced = sx_produced
        Reactor1.state = "active"
        return cos_produced

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
    ku = 1/pph
    kd = 1/pph
    
    def __init__(self):
        self.state = "idle"
        
    @classmethod
    def ss_output(cls, energy):
        return 1/(1 + math.exp(-energy + 3))-0.05
    
    def react(cls, energy, prev):
        Reactor2.state = "active"
        e_t = Reactor2.ss_output(energy)-prev
        if e_t > 0:
            return prev + Reactor2.ku*e_t
        else:
            return prev + Reactor2.kd*e_t
        
class Sx_filter:
    sat_factor = 3 # Sx filter saturation speed