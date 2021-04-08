from itertools import product

import dask.distributed
import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
from tqdm import tqdm

# src = tev_src
# dst = tev_dst 

src = tev_src = mkdir(Path("/Volumes/dedomeno/covid/vax-nature/focus_states_epi_1000_Apr01"))
dst = tev_dst = mkdir(Path("/Volumes/dedomeno/covid/vax-nature/focus_states_tev_1000_Apr01"))

# coefficients of consumption ~ prevalence regression
coeffs = pd.read_stata(data/"reg_estimates_india_TRYTHIS.dta")\
    [["parm", "estimate", "state_api", "district_api"]]\
    .rename(columns = {"parm": "param", "state_api": "state", "district_api": "district"})\
    .set_index("param")

I_coeff, D_coeff, constant = coeffs.loc[["I", "D", "_cons"]].estimate
month_FE = coeffs.filter(like = "month", axis = 0).estimate.values[
    pd.date_range(
        start   = simulation_start, 
        periods = simulation_range  + 1, 
        freq    = "D").month.values - 1
]
district_FE = coeffs.filter(like = "district", axis = 0)\
    .set_index(["state", "district"])["estimate"].to_dict()

# per capita daily consumption levels 
consumption_2019 = pd.read_stata("data/pcons_2019.dta")\
    .rename(columns = lambda _: _.replace("_api", ""))\
    .set_index(["state", "district"])

# life expectancy per state
years_life_remaining = pd.read_stata(data/"life_expectancy_2009_2013_collapsed.dta")\
    .assign(state = lambda _: _["state"].str.replace("&", "And"))\
    .set_index("state")\
    .rename(columns = {f"life_expectancy{i+1}": agebin_labels[i] for i in range(7)})

def rc_hat(state, district, dI_pc, dD_pc):
    """ estimate consumption decline """
    return (
        district_FE.get((state, district), 0) + constant + 
        month_FE[:, None] + 
        I_coeff * dI_pc   + 
        D_coeff * dD_pc
    )

def NPV(daily, n = simulation_range + 1, beta = 1/((1.0425)**(1/365))):
    """ calculate net present value over n periods at discount factor beta """
    s = np.arange(n)
    return [ 
        np.sum(np.power(beta, s[t:] - t)[:, None, None] * daily[t:, :], axis = 0)
        for t in range(n)
    ]

def counterfactual_metrics(q_p0v0, c_p0v0):
    """ evaluate health and econ metrics for the non-vaccination policy scenario """
    TEV_daily  = q_p0v0 * c_p0v0
    VSLY_daily = q_p0v0 * np.mean(c_p0v0, axis = 1)[:, None, :]
    return NPV(TEV_daily), NPV(VSLY_daily)

def policy_TEV(pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0):
    """ evaluate health and econ metrics for the vaccination policy scenario """
    # overall economic value
    TEV_daily = (1 - pi) * q_p1v0 * c_p1v0 + pi * q_p1v1 * c_p1v1 
    # health contribution to economic value
    dTEV_hlth = \
        (1 - pi) * (q_p1v0 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v0 +\
             pi  * (q_p1v1 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v1
    # consumption contribution to economic value
    dTEV_cons = \
        (1 - pi) * q_p0v0.mean(axis = 1)[:, None, :] * (c_p1v0 - c_p0v0.mean(axis = 1)[:, None, :]) +\
             pi  * q_p1v1.mean(axis = 1)[:, None, :] * (c_p1v1 - c_p0v0.mean(axis = 1)[:, None, :])
    # private contribution to economic value
    dTEV_priv = q_p1v1 * c_p1v1 - q_p1v0 * c_p1v0

    return (
        NPV(TEV_daily),
        NPV(dTEV_hlth, n = 1),
        NPV(dTEV_cons, n = 1),
        NPV(dTEV_priv, n = 1)
    )

def policy_VSLY(pi, q_p1v1, q_p1v0, c_p0v0):
        # value of statistical life year
    return NPV((((1 - pi) * q_p1v0) + (pi * q_p1v1)) * np.mean(c_p0v0, axis = 1)[:, None, :])

def save_metrics(name, metrics, dst = tev_dst):
    np.savez_compressed(dst/f"{name}.npz", metrics)

def process(district_data):
    """ run and save policy evaluation metrics """
    (state, district), state_code, N_district, N_0, N_1, N_2, N_3, N_4, N_5, N_6 = district_data
    N_jk = np.array([N_0, N_1, N_2, N_3, N_4, N_5, N_6])
    age_weight = N_jk/(N_j)
    rc_hat_p1v1 = rc_hat(state, district, np.zeros((simulation_range + 1, 1)), np.zeros((simulation_range + 1, 1)))
    c_p1v1 = np.transpose(
        (1 + rc_hat_p1v1)[:, None] * consumption_2019.loc[state, district].values[:, None],
            [0, 2, 1]
    )

    state_years_life_remaining = years_life_remaining.get(state, default = years_life_remaining.mean(axis = 0))
    phi_p0 = int(phi_points[0] * 365 * 100)
    cf_tag = f"{state_code}_{district}_phi{phi_p0}_novax"
    with np.load(src/(cf_tag + ".npz")) as counterfactual:
        dI_pc_p0 = counterfactual['dT']/N_district
        dD_pc_p0 = counterfactual['dD']/N_district
        q_p0v0   = counterfactual["q0"]
        D_p0     = counterfactual["Dj"]

    rc_hat_p0v0 = rc_hat(state, district, dI_pc_p0, dD_pc_p0)
    c_p0v0 = np.transpose(
        (1 + rc_hat_p0v0) * consumption_2019.loc[state, district].values[:, None, None],
        [1, 2, 0]
    )
    
    TEV_p0, VSLY_p0 = counterfactual_metrics(q_p0v0, c_p0v0)
    save_metrics("deaths_" + cf_tag, (D_p0[-1] - D_p0[0]).sum(axis = 1))
    save_metrics("YLL_"    + cf_tag, (D_p0[-1] - D_p0[0]) @ state_years_life_remaining)
    save_metrics("per_capita_TEV"  + cf_tag,  TEV_p0)
    save_metrics("per_capita_VSLY" + cf_tag, VSLY_p0)
    save_metrics("total_TEV"  + cf_tag, N_jk *  TEV_p0)
    save_metrics("total_VSLY" + cf_tag, N_jk * VSLY_p0)

    for (phi, vax_policy) in product(
        [int(_*365*100) for _ in phi_points], 
        ["random"]#, "contact"]#, "mortality"]
    ):
        p1_tag = f"{state_code}_{district}_phi{phi}_{vax_policy}"
        with np.load(src/(p1_tag + ".npz")) as policy:
            dI_pc_p1 = policy['dT']/N_district
            dD_pc_p1 = policy['dD']/N_district
            pi       = policy['pi'] 
            q_p1v1   = policy['q1']
            q_p1v0   = policy['q0']
            D_p1     = policy["Dj"]
        rc_hat_p1v0 = rc_hat(state, district, dI_pc_p1, dD_pc_p1)
        c_p1v0 = np.transpose(
            (1 + rc_hat_p1v0) * consumption_2019.loc[state, district].values[:, None, None], 
            [1, 2, 0]
        )

        TEV_p1, *_ = policy_TEV( pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0)
        VSLY_p1    = policy_VSLY(pi, q_p1v1, q_p1v0, c_p0v0)
        save_metrics("deaths_"          + p1_tag, (D_p1[-1] - D_p1[0]).sum(axis = 1))
        save_metrics("YLL_"             + p1_tag, (D_p1[-1] - D_p1[0]) @ state_years_life_remaining)
        save_metrics("per_capita_TEV_"  + p1_tag,  TEV_p1)
        save_metrics("per_capita_VSLY_" + p1_tag, VSLY_p1)
        save_metrics("total_TEV_"       + p1_tag,  TEV_p1 * N_jk)
        save_metrics("total_VSLY_"      + p1_tag, VSLY_p1 * N_jk)

if __name__ == "__main__":
    population_columns = ["state_code", "N_tot", 'N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6']
    distribute = True

    if distribute:
        with dask.config.set({"scheduler.allowed-failures": 5}):
            client = dask.distributed.Client()#(n_workers = 1, processes = False)
            print(client.dashboard_link)
            with dask.distributed.get_task_stream(client) as ts:
                futures = []
                # for district in districts_to_run[~districts_to_run.index.isin(["Andaman And Nicobar Islands"], level = 0)][population_columns].itertuples():
                for district in districts_to_run[districts_to_run.index.isin(["Tamil Nadu"], level = 0)][population_columns].itertuples():
                    futures.append(client.submit(process, district, key = ":".join(district[0])))
            dask.distributed.progress(futures)
    else:
        for t in tqdm(districts_to_run[districts_to_run.index.isin(["Ariyalur"], level = 1)][population_columns].itertuples()):
            process(t)
