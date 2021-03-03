from abc import abstractmethod
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.shape_base import dsplit
from scipy.stats import multinomial as Multinomial
from sklearn.metrics import auc

from .models import SIR
from .utils import weeks


def fillna(array):
    return np.nan_to_num(array, nan = 0, posinf = 0, neginf = 0)

def normalize(array, axis = 0):
    return fillna(array/array.sum(axis = axis))

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

# class VaccinationPolicy():
#     def __init__(self, daily_doses: int, effectiveness: float, S_bins: np.array, I_bins: np.array, age_ratios: np.array, IFRs: np.array) -> None:
#         self.daily_doses   = daily_doses 
#         self.effectiveness = effectiveness
#         self.S_bins        = S_bins.astype(float)
#         self.I_bins        = I_bins.astype(float)
#         self.age_ratios    = age_ratios
#         self.IFRs          = IFRs
#         self.dD_bins       = []

#     def name(self) -> str:
#         return self.__class__.__name__.lower()

#     def exhausted(self, model) -> bool:
#         return self.daily_doses * len(model.Rt) > model.pop0

#     @abstractmethod
#     def update_S_bins(self, immunizing_doses):
#         pass 

#     def distribute_doses(self, model: SIR, num_sims: int = 10_000) -> np.array:
#         if self.exhausted(model):
#             return (np.zeros(self.age_ratios.shape), np.zeros(self.age_ratios.shape), np.zeros(self.age_ratios.shape))
#         model.parallel_forward_epi_step(num_sims = num_sims)
#         num_immunizing_doses = ((model.S[-1]/model.N[-1]) * self.daily_doses * self.effectiveness)
#         model.S[-1] -= num_immunizing_doses

#         dI_bin_update = fillna(model.dT[-1][:, None] * self.S_bins/self.S_bins.sum())
#         self.S_bins -= dI_bin_update
#         self.I_bins += dI_bin_update
#         self.update_S_bins(num_immunizing_doses)

#         dD_bin_update = fillna(model.dD[-1][:, None] * (self.I_bins/self.I_bins.sum(axis = 1)[:, None]))
#         dR_bin_update = fillna(model.dR[-1][:, None] * (self.I_bins/self.I_bins.sum(axis = 1)[:, None]))
#         self.I_bins = (self.I_bins - (dD_bin_update + dR_bin_update)).clip(0)
#         self.dD_bins.append(dD_bin_update)
#         return num_immunizing_doses

#     def update_mortality(self) -> float:
#         return fillna((self.IFRs * self.I_bins/self.I_bins.sum(axis = 1)[:, None]).sum(axis = 1))

# class RandomVaccineAssignment(VaccinationPolicy):
#     def name(self) -> str:
#         return "randomassignment"

#     def update_S_bins(self, num_doses):
#         self.S_bins -= (self.S_bins/self.S_bins.sum(axis = 1)[:, None] * num_doses[:, None])

# class PrioritizedAssignment(VaccinationPolicy):
#     def __init__(self, daily_doses: int, effectiveness: float, S_bins: np.array, I_bins: np.array, age_ratios: np.array, IFRs: np.array, prioritization: List[int], label: str):
#         super().__init__(daily_doses, effectiveness, S_bins, I_bins, age_ratios, IFRs)
#         self.prioritization = prioritization
#         self.label          = label

#     def name(self) -> str:
#         return f"{self.label}prioritized"

#     def update_S_bins(self, num_doses):
#         # permute by prioritization to assign more easily 
#         self.S_bins = self.S_bins[:, self.prioritization].copy()
#         # find bins exhausted by latest dose
#         new_S_bins = np.where(self.S_bins.cumsum(axis = 1) <= num_doses[:, None], 0, self.S_bins) 
#         # subtract leftover doses from non-exhausted bins
#         new_S_bins[np.arange(len(new_S_bins)), (new_S_bins != 0).argmax(axis = 1)] -=\
#             np.squeeze(num_doses - np.where(self.S_bins.cumsum(axis = 1) > num_doses[:, None], 0, self.S_bins).sum(axis = 1))
#         # reverse permutation once assignment is done 
#         self.S_bins = new_S_bins[:, self.prioritization]

class VaccinationPolicy():
    def __init__(self, S_bins: np.array) -> None:
        self.S_bins = S_bins

    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def distribute_doses(self, model: SIR, *kwargs) -> Tuple[np.array]:
        pass 

    def exhausted(self, model) -> bool:
        return self.daily_doses * len(model.Rt) > model.pop0

    def get_mortality(self, base_IFRs) -> float:
        return base_IFRs @ normalize(self.S[-1] + self.S_vn[-1] + self.S_vm[-1])

class RandomVaccineAssignment(VaccinationPolicy):
    def __init__(self, daily_doses: int, effectiveness: float, 
        S_bins: np.array, 
        I_bins: np.array, 
        D_bins: np.array, 
        R_bins: np.array, 
        N_bins: np.array,
        age_ratios: np.array):
        self.daily_doses = daily_doses 
        self.age_ratios = age_ratios
        self.effectiveness = effectiveness
        
        self.N = [N_bins]
        self.S = [S_bins]
        self.I = [I_bins]
        self.R = [R_bins]
        self.D = [D_bins]

        shape = S_bins.shape

        self.S_vm = [np.zeros(shape)]
        self.S_vn = [np.zeros(shape)]
        self.R_vm = [np.zeros(shape)]
        self.R_vn = [np.zeros(shape)]
        self.I_vn = [np.zeros(shape)]
        self.D_vn = [np.zeros(shape)]

        self.N_v  = [np.zeros(shape)]
        self.N_nv = [np.zeros(shape)]

        self.pi   = [np.zeros(shape)]
        self.q_1  = [np.zeros(shape)]
        self.q_0  = [np.zeros(shape)]

    def distribute_doses(self, model: SIR, fS, fI, fR, fD, num_sims: int = 10000):
        model.parallel_forward_epi_step(num_sims = num_sims)

        doses = 0 if self.exhausted(model) else Multinomial.rvs(self.daily_doses, self.age_ratios)

        N_avg = model.N[-1].mean()

        dS_vm = doses * model.S[-1].mean() / N_avg * (    self.effectiveness)
        dS_vn = doses * model.S[-1].mean() / N_avg * (1 - self.effectiveness)
        dR_vm = doses * model.R[-1].mean() / N_avg
        
        dI_vn = doses * model.I[-1].mean() / N_avg
        
        dD_vn = model.dD[-1].mean() * normalize(self.I_vn[-1] + dI_vn)
        dR_vn = model.dR[-1].mean() * normalize(self.I_vn[-1] + dI_vn)

        model.S[-1] -= dS_vm.sum()

        self.S.append(self.S[-1] * fS[:, 0] - dS_vm - dS_vn)
        self.S_vm.append(self.S_vm[-1] + dS_vm)
        self.S_vn.append(self.S_vn[-1] + dS_vn)

        self.I.append(model.I[-1].mean() * normalize(self.S[-1] + dS_vm + dS_vn) - dI_vn)
        self.I_vn.append(self.I_vn[-1] + dI_vn)

        self.R.append(model.R[-1].mean() * fR[:, 0] - dR_vm - dR_vn)
        self.R_vm.append(self.R_vm[-1] + dR_vm)
        self.R_vn.append(self.R_vn[-1] + dR_vn)

        self.D.append(model.D[-1].mean() * fD[:, 0] - dD_vn)
        self.D_vn.append(self.D_vn[-1] + dD_vn)

        self.N_v.append(sum(_[-1] for _ in [self.S_vm, self.S_vn, self.I_vn, self.D_vn, self.R_vn, self.R_vm]))
        self.N_nv.append(self.N[0] - self.N_v[-1])

        self.pi.append(self.N_v[-1]/self.N[0])
        self.q_1.append((self.N_v[-1]  - self.D_vn[-1])/self.N_v[-1])
        self.q_0.append((self.N_nv[-1] - self.D[-1])/self.N_nv[-1])

    def name(self) -> str:
        return "randomassignment"

class PrioritizedAssignment(VaccinationPolicy):
    def __init__(self, daily_doses: int, effectiveness: float, 
        S_bins: np.array, 
        I_bins: np.array, 
        D_bins: np.array, 
        R_bins: np.array, 
        N_bins: np.array,
        age_ratios: np.array,
        prioritization: List[int], label: str):
        self.daily_doses = daily_doses 
        self.age_ratios = age_ratios
        self.effectiveness = effectiveness
        
        self.S_tot = [S_bins]
        self.N = [N_bins]
        self.S = [S_bins]
        self.I = [I_bins]
        self.R = [R_bins]
        self.D = [D_bins]

        shape = S_bins.shape

        self.S_vm = [np.zeros(shape)]
        self.S_vn = [np.zeros(shape)]
        self.R_vm = [np.zeros(shape)]
        self.R_vn = [np.zeros(shape)]
        self.I_vn = [np.zeros(shape)]
        self.D_vn = [np.zeros(shape)]

        self.N_v  = [np.zeros(shape)]
        self.N_nv = [np.zeros(shape)]

        self.pi   = [np.zeros(shape)]
        self.q_1  = [np.zeros(shape)]
        self.q_0  = [np.zeros(shape)]

        self.prioritization = prioritization
        self.label = label 

    def name(self) -> str:
        return f"{self.label}prioritized"

    def distribute_doses(self, model: SIR, fS, fI, fR, fD, num_sims: int = 10000):
        model.parallel_forward_epi_step(num_sims = num_sims)

        num_doses = 0 if self.exhausted(model) else self.daily_doses

        dVx_adm = np.zeros(self.S_tot[-1].shape)
        dVx_eff = np.zeros(self.S_tot[-1].shape)
        dVx_imm = np.zeros(self.S_tot[-1].shape)
        if num_doses > 0:
            bin_idx, age_bin = next(((i, age_bin) for (i, age_bin) in enumerate(self.prioritization) if self.S_tot[-1][age_bin] > 0), (None, None))
            if age_bin is not None:
                if self.S_tot[-1][age_bin] > num_doses:
                    dVx_adm[age_bin] = num_doses
                    dVx_eff[age_bin] = self.effectiveness * dVx_adm[age_bin]
                    dVx_imm[age_bin] = self.S_tot[-1][age_bin]/self.N[0][age_bin] * dVx_eff[age_bin]
                    self.S_tot[-1][age_bin] -= dVx_imm[age_bin]
                else: 
                    leftover = num_doses - self.S_tot[-1][age_bin]
                    dVx_adm[age_bin] = self.S_tot[-1][age_bin]
                    dVx_eff[age_bin] = self.effectiveness * dVx_adm[age_bin]
                    dVx_imm[age_bin] = self.S_tot[-1][age_bin]/self.N[0][age_bin] * dVx_eff[age_bin]
                    self.S_tot[-1][age_bin] -= dVx_imm[age_bin]
                    if bin_idx != len(self.S_tot[-1]) - 1:
                        dVx_adm[self.prioritization[bin_idx + 1]] = leftover
                        dVx_eff[self.prioritization[bin_idx + 1]] = leftover * self.effectiveness
                        dVx_imm[self.prioritization[bin_idx + 1]] = leftover * self.effectiveness * self.S_tot[-1][self.prioritization[bin_idx + 1]]/self.N[0][self.prioritization[bin_idx + 1]]
                        self.S_tot[-1][self.prioritization[bin_idx + 1]] -= dVx_imm[self.prioritization[bin_idx + 1]] 
            else: 
                print("vaccination exhausted", self.S_bins, self.prioritization)

        dS_vm = dVx_imm
        dS_vn = dVx_imm * (1 - self.effectiveness)/self.effectiveness

        dR_vm = (dVx_adm - dVx_imm) * model.R[-1].mean()/model.N[-1].mean()
        dI_vn = (dVx_adm - dVx_imm) * model.I[-1].mean()/model.N[-1].mean()
        
        dD_vn = model.dD[-1].mean() * normalize(self.I_vn[-1] + dI_vn)
        dR_vn = model.dR[-1].mean() * normalize(self.I_vn[-1] + dI_vn)

        self.S.append(self.S[-1] * fS[:, 0] - dS_vm - dS_vn)
        self.S_vm.append(self.S_vm[-1] + dS_vm)
        self.S_vn.append(self.S_vn[-1] + dS_vn)

        self.I.append(model.I[-1].mean() * normalize(self.S[-1] + self.S_vn[-1] + self.S_vm[-1]) - dI_vn)
        self.I_vn.append(self.I_vn[-1] + dI_vn)

        self.R.append(model.R[-1].mean() * fR[:, 0] - dR_vm - dR_vn)
        self.R_vm.append(self.R_vm[-1] + dR_vm)
        self.R_vn.append(self.R_vn[-1] + dR_vn)

        self.D.append(model.D[-1].mean() * fD[:, 0] - dD_vn)
        self.D_vn.append(self.D_vn[-1] + dD_vn)

        self.N_v.append(sum(_[-1] for _ in [self.S_vm, self.S_vn, self.I_vn, self.D_vn, self.R_vn, self.R_vm]))
        self.N_nv.append(self.N[0] - self.N_v[-1])

        self.pi.append(self.N_v[-1]/self.N[0])
        self.q_1.append((self.N_v[-1]  - self.D_vn[-1])/self.N_v[-1])
        self.q_0.append((self.N_nv[-1] - self.D[-1])/self.N_nv[-1])

        model.S[-1] -= dS_vm.sum()
