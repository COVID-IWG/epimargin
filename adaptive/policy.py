from typing import Callable, Dict, Optional
from itertools import product

import numpy as np

from .models import SIR
from .utils import days, weeks

from sklearn.metrics import auc 

def AUC(curve):
    return auc(x = range(len(curve)), y = curve)

def simulate_lockdown(model: SIR, lockdown_period: int, total_time: int, Rt0_mandatory: Dict[str, float], Rt0_voluntary: Dict[str, float], lockdown: np.matrix, migrations: np.matrix) -> SIR:
    return model.set_parameters(Rt0 = Rt0_mandatory)\
        .run(lockdown_period,  migrations = lockdown)\
        .set_parameters(Rt0 = Rt0_voluntary)\
        .run(total_time - lockdown_period, migrations = migrations)

def simulate_adaptive_control(
    model: SIR, 
    initial_run: int, 
    total_time: int, 
    lockdown: np.matrix, 
    migrations: np.matrix, 
    R_m: Dict[str, float],
    beta_v: Dict[str, float], 
    beta_m: Dict[str, float], 
    evaluation_period: int = 2*weeks, 
    adjacency: Optional[np.matrix] = None) -> SIR:
    n = len(model)
    model.set_parameters(Rt0 = R_m)\
         .run(initial_run, lockdown)
    days_run = initial_run
    gantt = []
    last_category = dict()
    while days_run < total_time:
        Gs, Ys, Os, Rs = set(), set(), set(), set()
        categories = dict(enumerate([Gs, Ys, Os, Rs]))
        category_transitions = {}
        for (i, unit) in enumerate(model):
            latest_Rt = unit.Rt[-1]
            if latest_Rt < 1: 
                Gs.add(i)
                beta_cat = 0
            else: 
                if days_run < initial_run + evaluation_period: # force first period to be lockdown
                    Rs.add(i)
                    beta_cat = 3
                else: 
                    if latest_Rt < 1.5: 
                        Ys.add(i)
                        beta_cat = 1
                    elif latest_Rt < 2: 
                        Os.add(i)
                        beta_cat = 2
                    else:
                        Rs.add(i)
                        beta_cat = 3
            if unit.name not in last_category:
                last_category[unit.name] = beta_cat
            else: 
                old_beta_cat = last_category[unit.name]
                if old_beta_cat != beta_cat:
                    if beta_cat < old_beta_cat and beta_cat != (old_beta_cat - 1): # force gradual release
                        beta_cat = old_beta_cat - 1
                        if i in categories[old_beta_cat]: categories[old_beta_cat].remove(i)
                        categories[beta_cat].add(i)
                    category_transitions[unit.name] = beta_cat
                    last_category[unit.name] = beta_cat 
            gantt.append([unit.name, days_run, beta_cat, max(0, latest_Rt)])

        for (unit_name, beta_cat) in category_transitions.items(): 
            unit =  model[unit_name]
            new_beta = beta_v[unit.name] - (beta_cat * (beta_v[unit.name] - beta_m[unit.name])/3.0)                
            unit.beta[-1] = new_beta
            unit.Rt0 = new_beta * unit.gamma

        phased_migration = migrations.copy()
        for (i, j) in product(range(n), range(n)):
            if i not in Gs or j not in Gs:
                phased_migration[i, j] = 0
        model.run(evaluation_period, phased_migration)
        days_run += evaluation_period

    model.gantt = gantt 
    return model 


def simulate_adaptive_control_MHA(model: SIR, initial_run: int, total_time: int, lockdown: np.matrix, migrations: np.matrix, R_m: Dict[str, float], beta_v: Dict[str, float], beta_m: Dict[str, float], evaluation_period = 2*weeks):
    n = len(model)
    model.set_parameters(Rt0 = R_m)\
         .run(initial_run, lockdown)
    days_run = initial_run
    gantt = []
    last_category = dict()
    while days_run < total_time:
        Gs, Ys, Os, Rs = set(), set(), set(), set()
        categories = dict(enumerate([Gs, Ys, Os, Rs]))
        category_transitions = {}
        for (i, unit) in enumerate(model):
            latest_Rt = unit.Rt[-1]
            if days_run < initial_run + evaluation_period: # force first period to MHA
                if unit.I[-4] != 0 and unit.I[-1]/unit.I[-4] > 2:
                    Rs.add(i)
                    beta_cat = 3
                else:
                    Gs.add(i)
                    beta_cat = 0
            else: 
                if latest_Rt < 1: 
                    Gs.add(i)
                    beta_cat = 0
                elif latest_Rt < 1.5: 
                    Ys.add(i)
                    beta_cat = 1
                elif latest_Rt < 2: 
                    Os.add(i)
                    beta_cat = 2
                else:
                    Rs.add(i)
                    beta_cat = 3
            if unit.name not in last_category:
                last_category[unit.name] = beta_cat
            else: 
                old_beta_cat = last_category[unit.name]
                if old_beta_cat != beta_cat:
                    if beta_cat < old_beta_cat and beta_cat != (old_beta_cat - 1): # force gradual release
                        beta_cat = old_beta_cat - 1
                    if i in categories[old_beta_cat]: categories[old_beta_cat].remove(i)
                    categories[beta_cat].add(i)
                    category_transitions[unit.name] = beta_cat
                    last_category[unit.name] = beta_cat 
            gantt.append([unit.name, days_run, beta_cat, max(0, latest_Rt)])

        for (unit_name, beta_cat) in category_transitions.items(): 
            unit =  model[unit_name]
            new_beta = beta_v[unit.name] - (beta_cat * (beta_v[unit.name] - beta_m[unit.name])/3.0)                
            unit.beta[-1] = new_beta
            unit.Rt0 = new_beta * unit.gamma

        phased_migration = migrations.copy()
        for (i, j) in product(range(n), range(n)):
            if i not in Gs or j not in Gs:
                phased_migration[i, j] = 0
        model.run(evaluation_period, phased_migration)
        days_run += evaluation_period

    model.gantt = gantt 
    return model 

def simulate_PID_controller(
    model: SIR, 
    initial_run: int, 
    total_time: int,
    Rtarget: float = 0.9,
    kP: float = 0.05, 
    kI: float = 0.5,
    kD: float = 0,
    Dt: float = 1.0) -> SIR:
    # initial run without PID
    model.run(initial_run)
    
    # set up PID running variables
    integral   = 0
    derivative = 0
    u = 0
    prev_error = model[0].Rt[-1]

    z = np.zeros((len(model), len(model)))

    # run forward model 
    for i in range(total_time - initial_run):
        model[0].Rt0 -= u 
        model.run(1, z)

        error = model[0].Rt[-1] - Rtarget
        integral  += error * Dt 
        derivative = (error - prev_error)/Dt

        u = kP * error + kI * integral + kD * derivative
        prev_error = error
    return model 

def vaccination_policy(
    model: SIR, 
    age_structure: np.array, 
    total_vaccines: int, 
    vaccination_rate: float, 
    vaccination_effectiveness: float,
    prioritization: Callable[[SIR, np.array]],
    t: Optional[int] = None
):
    pass