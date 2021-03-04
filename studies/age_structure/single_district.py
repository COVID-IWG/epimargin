from itertools import product

import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import Age_SIRVD
from adaptive.policy import PrioritizedAssignment, RandomVaccineAssignment, VaccinationPolicy
from studies.age_structure.commons import *
from studies.age_structure.palette import *

from scipy.stats import multinomial
from adaptive.utils import normalize

sns.set(style = "whitegrid")

def update_mortality(model, IFRs):
    return normalize(model.I[-1] + model.I_vn[-1]) @ IFRs

num_sims = 1

district = "Chennai"
seroprevalence = 0.4094
N_district = district_populations[district]

D0, R0 = ts.loc[district][["dD", "dR"]].sum()

dT_conf_district = ts.loc[district].dT
dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
T_conf_smooth = dT_conf_district_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N_district * seroprevalence)
T_ratio = T_sero/T

T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
S0 = N_district - T_scaled
dD0 = ts.loc[district].dD.loc[simulation_start]
I0 = max(0, (T_scaled - R0 - D0))

(Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

random_vax_model = Age_SIRVD(
    name        = district, 
    population  = N_district - D0, 
    dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
    Rt0         = Rt[simulation_start],
    S0          = (fS * S0).T, 
    I0          = (fI * I0).T, 
    R0          = (fR * R0).T, 
    D0          = (fD * D0).T,
    mortality   = (ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
    random_seed = 0
)

phi = 0.25/365

no_vax_model = Age_SIRVD(
    name        = district, 
    population  = N_district, 
    dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
    Rt0         = Rt[simulation_start],
    S0          = (fS * S0).T, 
    I0          = (fI * I0).T, 
    R0          = (fR * R0).T, 
    D0          = (fD * D0).T,
    mortality   = (ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
    random_seed = 0
)
dVs = []
try: 
    for _ in range(1 * 365):
        # if phi * (S0 + I0 + R0) * _ >= random_vax_model.N[0].sum():
        #     dV = np.zeros((7, num_sims))
        # else: 
        dV = multinomial.rvs(phi * N_district, normalize(random_vax_model.N[-1], axis = 1)[0]).reshape((-1, num_sims))
        dVs.append(dV)
        random_vax_model.parallel_forward_epi_step(dV, num_sims = 1)
        m = update_mortality(random_vax_model, list(TN_IFRs.values()) )
        random_vax_model.m = m
        # print(_, "S   ", random_vax_model.S   [-1].round(1))
        # print(_, "S_vm", random_vax_model.S_vm[-1].round(1))
        # print(_, "S_vn", random_vax_model.S_vn[-1].round(1))
        # print(_, "I   ", random_vax_model.I   [-1].round(1))
        # print(_, "I_vn", random_vax_model.I_vn[-1].round(1))
        # print(_, "R   ", random_vax_model.R   [-1].round(1))
        # print(_, "R_vm", random_vax_model.R_vm[-1].round(1))
        # print(_, "R_vn", random_vax_model.R_vn[-1].round(1))
        # print(_, "D   ", random_vax_model.D   [-1].round(1))
        # print(_, "D_vn", random_vax_model.D_vn[-1].round(1))
        
        no_vax_model.parallel_forward_epi_step(dV = np.zeros((7, num_sims)), num_sims = 1)
        no_vax_model.m = update_mortality(no_vax_model, list(TN_IFRs.values()))
        # print(random_vax_model.m, no_vax_model.m)
except: 
    self = random_vax_model 
    S, S_vm, S_vn, I, I_vn, R, R_vm, R_vn, D, D_vn, N = (_[-1].copy() for _ in 
        (self.S, self.S_vm, self.S_vn, self.I, self.I_vn, self.R, self.R_vm, self.R_vn, self.D, self.D_vn, self.N))
    from adaptive.utils import fillna, normalize
    from scipy.stats import poisson
    raise 

print("random D", random_vax_model.D[-1].sum())
print("no vax D",     no_vax_model.D[-1].sum())
print("random D since vax start", random_vax_model.D[-1].sum() - random_vax_model.D[0].sum())
print("no vax D since vax start",     no_vax_model.D[-1].sum() -     no_vax_model.D[0].sum())

print("random q1", random_vax_model.q1[-1].round(4))
print("random q0", random_vax_model.q0[-1].round(4))
print("no vax q0", no_vax_model.q0[-1].round(4))

self = random_vax_model 
S, S_vm, S_vn, I, I_vn, R, R_vm, R_vn, D, D_vn, N = (_[-1].copy() for _ in 
    (self.S, self.S_vm, self.S_vn, self.I, self.I_vn, self.R, self.R_vm, self.R_vn, self.D, self.D_vn, self.N))