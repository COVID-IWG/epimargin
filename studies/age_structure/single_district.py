
from itertools import product
import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import Age_SIRVD
from adaptive.policy import (PrioritizedAssignment, RandomVaccineAssignment,
                             VaccinationPolicy)
from adaptive.utils import normalize
from scipy.stats import multinomial
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from studies.age_structure.wtp import *

sns.set(style = "whitegrid")

def get_WTP(label, policy, counterfactual, district = "Chennai"):
    # print(f"{label}  D", policy.D[-1].sum())
    # print("no vax    D", counterfactual.D[-1].sum())
    # print(f"{label} since vax start", (policy.D[-1] - policy.D[0]).sum())
    # print("no vax D since vax start", (counterfactual.D[-1] - counterfactual.D[0]).sum())
    # print(f"{label} q1", policy.q1[-1])
    # print(f"{label} q0", policy.q0[-1])
    # print("no vax q0", counterfactual.q0[-1])
    # print("sign(dq0)", np.sign(policy.q0[-1] - counterfactual.q0[-1]))

    f_hat_p1v1 = estimate_consumption_decline(district, 
        pd.Series(np.zeros(366)), 
        pd.Series(np.zeros(366)), force_natl_zero = True)
    c_p1v1 = (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values

    dI_pc_p1 = pd.DataFrame(np.sum(np.squeeze(policy.I) + np.squeeze(policy.I_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    dD_pc_p1 = pd.DataFrame(np.sum(np.squeeze(policy.D) + np.squeeze(policy.D_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    f_hat_p1v0 = estimate_consumption_decline(district, dI_pc_p1, dD_pc_p1)
    c_p1v0 = (1 + f_hat_p1v0)[:, None] * consumption_2019.loc[district].values

    dI_pc_p0 = pd.Series(np.sum(np.squeeze(counterfactual.I) + np.squeeze(counterfactual.I_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    dD_pc_p0 = pd.Series(np.sum(np.squeeze(counterfactual.D) + np.squeeze(counterfactual.D_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    f_hat_p0v0 = estimate_consumption_decline(district, dI_pc_p0, dD_pc_p0)
    c_p0v0 = (1 + f_hat_p0v0)[:, None] * consumption_2019.loc[district].values

    pi = np.squeeze(policy.pi) 
    q_p1v1 = np.squeeze(policy.q1)
    q_p1v0 = np.squeeze(policy.q0)

    WTP_daily_1 = pi * q_p1v1 * c_p1v1 + (1 - pi) * q_p1v0 * c_p1v0
    WTP_daily_0 = q_p1v0 * c_p0v0

    beta = 1/(1 + 4.25/365)
    s = np.arange(366)

    dWTP_daily = WTP_daily_1 - WTP_daily_0

    WTPs = []
    for t in range(366):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None] * dWTP_daily[t:, :], axis = 0)
        WTPs.append(wtp)
    WTPs = np.squeeze(WTPs)

    plt.plot(dWTP_daily)
    plt.title(f"$\Delta$WTP - {label}")
    plt.figure()
    plt.plot(WTPs)
    plt.title(f"WTP($t$) - {label}")
    plt.figure()
    plt.show()

    return dWTP_daily, WTPs


def prioritize(num_doses, S, prioritization):
    Sp = S[:, prioritization]
    dV = np.where(Sp.cumsum(axis = 1) <= num_doses, Sp, 0)
    dV[np.arange(len(dV)), (Sp.cumsum(axis = 1) > dV.cumsum(axis = 1)).argmax(axis = 1)] = num_doses - dV.sum(axis = 1)
    return dV[:, [i for (i, _) in sorted(enumerate(prioritization), key = lambda t: t[1])]]

num_sims = 2
num_age_bins = 7
district = "Chennai"
seroprevalence = 0.4094
N_district = district_populations[district]

across_bins = dict(axis = 0)
across_time = dict(axis = 1)

dT_conf_district = ts.loc[district].dT
dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
T_conf_smooth = dT_conf_district_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N_district * seroprevalence)
T_ratio = T_sero/T
D0, R0 = ts.loc[district][["dD", "dR"]].sum()
R0 *= T_ratio

T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
S0 = N_district - T_scaled
dD0 = ts.loc[district].dD.loc[simulation_start]
I0 = max(0, (T_scaled - R0 - D0))

(Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

def get_model(seed = 104):
    return Age_SIRVD(
        name        = district, 
        population  = N_district - D0, 
        dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
        Rt0         = Rt[simulation_start],
        S0          = np.tile((fS * S0).T, num_sims).reshape((num_sims, -1)),
        I0          = np.tile((fI * I0).T, num_sims).reshape((num_sims, -1)),
        R0          = np.tile((fR * R0).T, num_sims).reshape((num_sims, -1)),
        D0          = np.tile((fD * D0).T, num_sims).reshape((num_sims, -1)),
        mortality   = np.array(list(TN_IFRs.values())), #(ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
        random_seed = seed
    )


seed = 0
for phi in (0.25/365, 0.5/365):
    num_doses = phi * (S0 + I0 + R0)

    random_model, mortality_model, contact_model, no_vax_model = [get_model(seed) for _ in range(4)]

    dVs = {_:[] for _ in ("random", "mortality", "contact")}

    try: 
        for t in range(1 * 365):
            if t <= 1/phi:
                dV_random    = multinomial.rvs(num_doses, normalize(random_model.N[-1], axis = 1)[0]).reshape((-1, 7))
                dV_mortality = prioritize(num_doses, mortality_model.S[-1], [6, 5, 4, 3, 2, 1, 0]) 
                dV_contact   = prioritize(num_doses, contact_model.S[-1], [1, 2, 3, 4, 0, 5, 6]) 
            else: 
                dV_random, dV_mortality, dV_contact = np.zeros((num_sims, 7)), np.zeros((num_sims, 7)), np.zeros((num_sims, 7))
            dVs["random"].append(dV_random)
            dVs["mortality"].append(dV_mortality)
            dVs["contact"].append(dV_contact)
            
            random_model.parallel_forward_epi_step(dV_random, num_sims = num_sims)
            mortality_model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)
            contact_model.parallel_forward_epi_step(dV_contact, num_sims = num_sims)
            no_vax_model.parallel_forward_epi_step(dV = np.zeros((7, num_sims))[:, 0], num_sims = num_sims)
    except: 
        self = random_model 
        S, S_vm, S_vn, I, I_vn, R, R_vm, R_vn, D, D_vn, N = (_[-1].copy() for _ in 
            (self.S, self.S_vm, self.S_vn, self.I, self.I_vn, self.R, self.R_vm, self.R_vn, self.D, self.D_vn, self.N))
        from adaptive.utils import fillna, normalize
        from scipy.stats import poisson
        raise 

self = random_model 
S, S_vm, S_vn, I, I_vn, R, R_vm, R_vn, D, D_vn, N = (_[-1].copy() for _ in 
    (self.S, self.S_vm, self.S_vn, self.I, self.I_vn, self.R, self.R_vm, self.R_vn, self.D, self.D_vn, self.N))

# get_WTP("mortality", mortality_model, no_vax_model)
# get_WTP("contact",   contact_model,   no_vax_model)
# get_WTP("random",    random_model,    no_vax_model)