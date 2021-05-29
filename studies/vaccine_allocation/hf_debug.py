from itertools import chain, product

import epimargin.plots as plt
from epimargin.etl.covid19india import state_code_lookup
from studies.vaccine_allocation.commons import *
from studies.vaccine_allocation.epi_simulations import *

from studies.vaccine_allocation.natl_figures import aggregate_static_percentiles, outcomes_per_policy, aggregate_dynamic_percentiles
from studies.vaccine_allocation.policy_evaluation import years_life_remaining


src = fig_src

phis = [int(_ * 365 * 100) for _ in phi_points]
params = list(chain([(phis[0], "novax",)], product(phis, ["contact", "random", "mortality"])))

population_columns = ["state_code", "N_tot", 'N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6', 'T_ratio']
state_code, N_district, N_0, N_1, N_2, N_3, N_4, N_5, N_6, T_ratio = districts_to_run.iloc[0][population_columns]
N_jk = np.array([N_0, N_1, N_2, N_3, N_4, N_5, N_6])

for s, label in [
    ("oo", "old VSLY, old q1"),
    ("o1", "old VSLY, q1 = 1"),
    ("on", "old VLSY, new q1"),
    ("no", "new VSLY, old q1"),
    ("nn", "new VSLY, new q1")
]:
    VSLY_percentiles = {
        p: aggregate_dynamic_percentiles(src, f"total_{s}VSLY_{state_code}_{district}_*phi{'_'.join(map(str, p))}.npz")
        for p in tqdm(params[1:])
    }
    outcomes_per_policy(
        {k: v * USD/(1e9) for (k, v) in VSLY_percentiles.items()}, "VSLY (USD, billions)", "D",
        reference = (25, "contact"), 
        phis = [25, 50, 100, 200, 500, 1000], 
        vax_policies = ["contact", "random", "mortality"], 
        policy_colors = [contactrate_vax_color, random_vax_color, mortality_vax_color], 
        policy_labels = ["contact rate", "random", "mortality"]
    )
    plt.PlotDevice().title(f"VSLY Gopalganj: {label}").show()


q_bar_sum = {}
for (phi, policy) in params[1:]:
    qbs = np.sum(np.load(src / f"q_bar_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    q_bar_sum[phi, policy] = np.percentile(qbs, [50, 5, 95], axis = 0) @ N_jk

outcomes_per_policy(q_bar_sum, reference = (25, "contact"), metric_label = "sum qbar", fmt = "D")



q_bar_sum_LE = {}
for (phi, policy) in params[1:]:
    qbs = np.sum(np.load(src / f"q_bar_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    q_bar_sum_LE[phi, policy] = np.percentile(qbs, [50, 5, 95], axis = 0) @ (N_jk * years_life_remaining.loc["Bihar"])
outcomes_per_policy(q_bar_sum_LE, reference = (25, "contact"), metric_label = "sum qbar * LE", fmt = "D")


v0_sum = {}
v0_raw = {}
for (phi, policy) in params[1:]:
    v0 = np.sum(np.load(src / f"v0_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    v0_raw[phi, policy] = np.sum(np.percentile(v0, [50, 5, 95], axis = 0), axis = -1)
    v0_sum[phi, policy] = np.percentile(v0, [50, 5, 95], axis = 0) @ N_jk
outcomes_per_policy(v0_raw, reference = (25, "contact"), metric_label = "v0 (no population weights)", fmt = "D")
outcomes_per_policy(v0_sum, reference = (25, "contact"), metric_label = "v0", fmt = "D")

v1_sum = {}
v1_raw = {}
for (phi, policy) in params[1:]:
    v1 = np.sum(np.load(src / f"v1_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    v1_raw[phi, policy] = np.sum(np.percentile(v1, [50, 5, 95], axis = 0), axis = -1)
    v1_sum[phi, policy] = np.percentile(v1, [50, 5, 95], axis = 0) @ N_jk
outcomes_per_policy(v1_raw, reference = (25, "contact"), metric_label = "v1 (no population weights)", fmt = "D")
outcomes_per_policy(v1_sum, reference = (25, "contact"), metric_label = "v1", fmt = "D")

v0_sum = {}
v0_raw = {}
for (phi, policy) in params[1:]:
    v0 = np.sum(np.load(src / f"v0_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    v0_raw[phi, policy] = np.sum(np.percentile(v0, [50, 5, 95], axis = 0), axis = -1)
    v0_sum[phi, policy] = np.percentile(v0, [50, 5, 95], axis = 0) @ N_jk
outcomes_per_policy(v0_raw, reference = (25, "contact"), metric_label = "v0 (no population weights)", fmt = "D")
outcomes_per_policy(v0_sum, reference = (25, "contact"), metric_label = "v0", fmt = "D")

v1_sum = {}
v1_raw = {}
for (phi, policy) in params[1:]:
    v1 = np.sum(np.load(src / f"v1_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0)
    v1_raw[phi, policy] = np.sum(np.percentile(v1, [50, 5, 95], axis = 0), axis = -1)
    v1_sum[phi, policy] = np.percentile(v1, [50, 5, 95], axis = 0) @ N_jk
outcomes_per_policy(v1_raw, reference = (25, "contact"), metric_label = "v1 (no population weights)", fmt = "D")
outcomes_per_policy(v1_sum, reference = (25, "contact"), metric_label = "v1", fmt = "D")

proxy_sum = {}
proxy_raw = {}
for (phi, policy) in params[1:]:
    pr = np.sum(np.load(src / f"proxyLLweight_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = (0, 2))
    ps = np.sum(np.load(src / f"proxyLLweight_BR_Gopalganj_phi{phi}_{policy}.npz")['arr_0'], axis = 0) @ (N_jk * years_life_remaining.loc["Bihar"])
    proxy_raw[phi, policy] = np.percentile(pr, [50, 5, 95], axis = 0)
    proxy_sum[phi, policy] = np.percentile(ps, [50, 5, 95], axis = 0)
outcomes_per_policy(proxy_raw, reference = (25, "contact"), metric_label = "sum 1-qbar * LE (no population weights)", fmt = "D")
# outcomes_per_policy(proxy_sum, reference = (25, "contact"), metric_label = "sum 1-qbar * LE", fmt = "D")

for phi, policy in params:
    plt.plot(
        np.median(
            np.load(tev_src / f"BR_Gopalganj_phi{phi}_{policy}.npz")['dT'], 
            axis = 1),
        label = (phi, policy)
    )
plt.legend()
plt.show()


for phi, policy in params:
    plt.plot(
        np.median(
            np.load(tev_src / f"BR_Gopalganj_phi{phi}_{policy}.npz")['dD'], 
            axis = 1),
        label = (phi, policy)
    )
plt.legend()
plt.show()

policy = "random"
for phi in phis:
    print(phi)
    pi = np.load(tev_src / f"BR_Gopalganj_phi{phi}_{policy}.npz")["pi"]
    q0 = np.load(tev_src / f"BR_Gopalganj_phi{phi}_{policy}.npz")["q0"]
    q1 = np.load(tev_src / f"BR_Gopalganj_phi{phi}_{policy}.npz")["q1"]

    plt.plot(np.median(pi, axis = 1)) 
    plt.show()
    plt.plot(np.median(q0, axis = 1)) 
    plt.show()
    plt.plot(np.median(q1, axis = 1)) 
    plt.show()