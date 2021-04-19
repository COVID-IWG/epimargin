from itertools import chain, product
from studies.age_structure.policy_evaluation import *

import epimargin.plots as plt

src = mkdir(Path("/Volumes/dedomeno/covid/vax-nature/focus_states_epi_100_Apr01"))
population_columns = ["state_code", "N_tot", 'N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6']

params = list(chain([("no_vax",)], product([25, 50, 100, 200], ["mortality"])))
state, district = "Tamil Nadu", "Chennai"

state_code, N_district, N_0, N_1, N_2, N_3, N_4, N_5, N_6 = districts_to_run.loc[state, district][population_columns]

def get_cons_decline(p1, p2 = None):
    if p2: 
        p = src/f"{state_code}_{district}_phi{p1}_{p2}.npz"
    else:
        p = src/f"{state_code}_{district}_phi25_novax.npz"
    with np.load(p) as policy:
        dI_pc = policy['dT']/N_district
        dD_pc = policy['dD']/N_district
    return income_decline(state, district, dI_pc, dD_pc)

rc_vectors = {p: np.median(get_cons_decline(*p), axis = 1) for p in params}

for (k, v) in rc_vectors.items():
    plt.plot(v, label = k)
plt.legend()
plt.show()

state_code = "TN"
TN_cons_predicted = {p:0 for p in params}
TN_cons2019 = consumption_2019.loc["Tamil Nadu"].sum(axis = 1)
rcs = dict()
for district in TN_cons2019.index:
    N_district = districts_to_run.loc["Tamil Nadu", district].N_tot
    for p in params:
        if len(p) > 1: 
            p1, p2 = p
            simfile = src/f"{state_code}_{district}_phi{p1}_{p2}.npz"
        else:
            simfile = src/f"{state_code}_{district}_phi25_novax.npz"
        with np.load(simfile) as policy:
            dI_pc = policy['dT']/N_district
            dD_pc = policy['dD']/N_district
        rc = income_decline(state, district, dI_pc, dD_pc)/100
        TN_cons_predicted[p] += N_district * (1 + rc) * TN_cons2019.loc[district]
        rcs[district, p] = rc

N_TN = districts_to_run.loc["Tamil Nadu"].N_tot.sum()
for (k, v) in TN_cons_predicted.items():
    plt.plot(np.median(v, axis = 1)/N_TN, label = k)
plt.legend()
plt.show()
