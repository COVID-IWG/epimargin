from abc import abstractmethod
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from pandas.core import base
from scipy.stats import multinomial as Multinomial
from sklearn.metrics import auc

from .models import SIR
from .utils import days, weeks

def fillna(array):
    return np.nan_to_num(array, nan = 0, posinf = 0, neginf = 0)

def AUC(curve):
    return auc(x = range(len(curve)), y = curve)

# Adaptive Control policies

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

# Vaccination policies

class VaccinationPolicy():
    def __init__(self, daily_doses: int, effectiveness: float, S_bins: np.array, I_bins: np.array, age_ratios: np.array, IFRs: np.array) -> None:
        self.daily_doses   = daily_doses 
        self.effectiveness = effectiveness
        self.S_bins        = S_bins.astype(int)
        self.I_bins        = I_bins.astype(int)
        self.age_ratios    = age_ratios
        self.IFRs          = IFRs
        self.dD_bins       = []

    def name(self) -> str:
        return self.__class__.__name__.lower()

    def exhausted(self, model) -> bool:
        return self.daily_doses * len(model.Rt) > model.pop0

    @abstractmethod
    def update_S_bins(self, immunizing_doses):
        pass 

    def distribute_doses(self, model: SIR, num_sims: int = 10_000) -> np.array:
        if self.exhausted(model):
            return (np.zeros(self.age_ratios.shape), np.zeros(self.age_ratios.shape), np.zeros(self.age_ratios.shape))
        model.parallel_forward_epi_step(num_sims = num_sims)
        num_immunizing_doses = ((model.S[-1]/model.N[-1]) * self.daily_doses * self.effectiveness).astype(int)
        model.S[-1] -= num_immunizing_doses

        dI_bin_update = fillna(model.dT[-1][:, None] * self.S_bins/self.S_bins.sum()).astype(int)
        self.S_bins -= dI_bin_update
        self.I_bins += dI_bin_update
        self.update_S_bins(num_immunizing_doses)

        dD_bin_update = fillna(model.dD[-1][:, None] * (self.I_bins/self.I_bins.sum(axis = 1)[:, None] * self.IFRs)).astype(int)
        dR_bin_update = fillna(model.dR[-1][:, None] * (self.I_bins/self.I_bins.sum(axis = 1)[:, None] * (1 - self.IFRs))).astype(int)
        self.I_bins = (self.I_bins - (dD_bin_update + dR_bin_update)).clip(0)
        self.dD_bins.append(dD_bin_update)
        return num_immunizing_doses

    def update_mortality(self) -> float:
        return fillna((self.IFRs * self.I_bins/self.I_bins.sum(axis = 1)[:, None]).sum(axis = 1))

class RandomVaccineAssignment(VaccinationPolicy):
    def name(self) -> str:
        return "randomassignment"

    def update_S_bins(self, num_doses):
        self.S_bins - (self.S_bins/self.S_bins.sum(axis = 1)[:, None] * num_doses[:, None]).astype(int)

class PrioritizedAssignment(VaccinationPolicy):
    def __init__(self, daily_doses: int, effectiveness: float, S_bins: np.array, I_bins: np.array, age_ratios: np.array, IFRs: np.array, prioritization: List[int], label: str):
        super().__init__(daily_doses, effectiveness, S_bins, I_bins, age_ratios, IFRs)
        self.prioritization = prioritization
        self.label          = label

    def name(self) -> str:
        return f"{self.label}prioritized"

    def update_S_bins(self, num_doses):
        # permute by prioritization to assign more easily 
        self.S_bins = self.S_bins[:, self.prioritization]
        # find bins exhausted by latest dose
        new_S_bins = np.where(self.S_bins.cumsum(axis = 1) <= num_doses[:, None], 0, self.S_bins) 
        # subtract leftover doses from non-exhausted bins
        new_S_bins[np.arange(len(new_S_bins)), (new_S_bins != 0).argmax(axis = 1)] -=\
            np.squeeze(num_doses - np.where(self.S_bins.cumsum(axis = 1) > num_doses[:, None], 0, self.S_bins).sum(axis = 1))
        # reverse permutation once assignment is done 
        self.S_bins = new_S_bins[:, self.prioritization]
