import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
import dask 

""" Calculate WTP, VSLY, and other policy evaluation metrics """ 

src = tev_src
dst = tev_dst
# dst = mkdir(Path("/Volumes/dedomeno/covid/vax-nature/focus_states_wrongcons_tev_100_Apr01"))


# coefficients of consumption ~ prevalence regression
coeffs = pd.read_stata(data/"reg_estimates_india_TRYTHIS.dta")\
    [["parm", "estimate", "state_api", "district_api"]]\
    .rename(columns = {"parm": "param", "state_api": "state", "district_api": "district"})\
    .set_index("param")

def nest_dict(paramkey, coeffs):
    return {
        k: v.to_dict() for (k, v) in coeffs\
        .filter(regex = f".*{paramkey}$", axis = 0)\
        .reset_index()\
        .assign(quantile = lambda _: _["param"]\
            .str.split(".")\
            .str[0]\
            .str.replace("b", "")\
            .str.replace("o", "")\
            .astype(int))\
        .set_index("quantile")\
        .groupby("state")\
        ["estimate"]
    } 
I_cat_coeffs = nest_dict("I_cat", coeffs)
D_cat_coeffs = nest_dict("D_cat", coeffs)
month_coeffs = nest_dict("month", coeffs)['']

district_coeffs = coeffs.set_index(["state", "district"])["estimate"].to_dict()

# life expectancy per state
YLLs = pd.read_stata(data/"life_expectancy_2009_2013_collapsed.dta")\
    .assign(state = lambda _: _["state"].str.replace("&", "And"))\
    .set_index("state")\
    .rename(columns = {f"life_expectancy{i+1}": agebin_labels[i] for i in range(7)})

# per capita daily consumption levels 
consumption_2019 = pd.read_stata("data/pcons_2019.dta")\
    .rename(columns = lambda _: _.replace("_api", ""))\
    .set_index(["state", "district"])

def dict_map(arr, mapping): 
    # from https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    mapping_ar = np.zeros(k.max()+1, dtype = v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[arr]

def income_decline(state, district, dI_pc, dD_pc):
    month = pd.date_range(start = simulation_start, periods = len(dI_pc), freq = "D").month.values

    return (
        coeffs.loc["I"]["estimate"] * dI_pc       + 
        coeffs.loc["D"]["estimate"] * dD_pc       + 
        dict_map(month, month_coeffs)[:, None]    + 
        district_coeffs.get((state, district), 0) + 
        coeffs.loc["_cons"]["estimate"] 
    )

def discounted_WTP(wtp, rate = (4.25/100), period = "daily"):
    if period == "daily":
        rate /= 365
    elif period == "monthly":
        rate /= 12 
    return (wtp * ((1 + rate) ** -np.arange(len(wtp)))[:, None]).sum(axis = 0)

def counterfactual_metrics(q_p0v0, c_p0v0):
    WTP_daily_0  = q_p0v0 * c_p0v0
    VSLY_daily_0 = q_p0v0 * np.mean(c_p0v0, axis = 1)[:, None, :]

    WTP_NPV  = [] 
    VSLY_NPV = []
    
    beta = 1/((1.0425)**(1/365))
    s = np.arange(simulation_range + 1)
    for t in range(simulation_range + 1):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * WTP_daily_0[t:, :], axis = 0)
        WTP_NPV.append(wtp)

        vsly = np.sum(np.power(beta, s[t:] - t)[:, None, None] * VSLY_daily_0[t:, :], axis = 0)
        VSLY_NPV.append(vsly)
    
    return WTP_NPV, VSLY_NPV

def get_metrics(pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0):  
    WTP_daily_1 = (1 - pi) * q_p1v0 * c_p1v0 + pi * q_p1v1 * c_p1v1 

    dWTP_health_daily = \
        (1 - pi) * (q_p1v0 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v0 +\
             pi  * (q_p1v1 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v1

    dWTP_income_daily = \
        (1 - pi) * q_p0v0.mean(axis = 1)[:, None, :] * (c_p1v0 - c_p0v0.mean(axis = 1)[:, None, :]) +\
             pi  * q_p1v1.mean(axis = 1)[:, None, :] * (c_p1v1 - c_p0v0.mean(axis = 1)[:, None, :])

    dWTP_private_daily = \
        q_p1v1 * c_p1v1 - q_p1v0 * c_p1v0

    vax_hazard   = ((1 - pi) * q_p1v0 + pi * q_p1v1)
    mean_cons =  np.mean(c_p0v0, axis = 1)[:, None, :]
    VSLY_daily_1 = ((1 - pi) * q_p1v0 + pi * q_p1v1) * mean_cons

    beta = 1/((1.0425)**(1/365))
    s = np.arange(simulation_range + 1)

    WTP_NPV  = [] 
    VSLY_NPV = []
    VSLY_no_discount = []
    WTP_health_NPV0  = None
    WTP_income_NPV0  = None
    WTP_private_NPV0 = None
    vh_discount = []
    for t in range(simulation_range + 1):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * WTP_daily_1[t:, :], axis = 0)
        WTP_NPV.append(wtp)

        vsly = np.sum(np.power(beta, s[t:] - t)[:, None, None] * VSLY_daily_1[t:, :], axis = 0)
        VSLY_NPV.append(vsly)
        vsly_beta1 = np.sum(np.power(1   , s[t:] - t)[:, None, None] * VSLY_daily_1[t:, :], axis = 0)
        VSLY_no_discount.append(vsly_beta1)

        vh_discount

        if t == 1:
            WTP_health_NPV0  = np.sum(np.power(beta, s - t)[:, None, None] * dWTP_health_daily , axis = 0)
            WTP_income_NPV0  = np.sum(np.power(beta, s - t)[:, None, None] * dWTP_income_daily , axis = 0)
            WTP_private_NPV0 = np.sum(np.power(beta, s - t)[:, None, None] * dWTP_private_daily, axis = 0) 

    return (
        WTP_NPV,
        WTP_health_NPV0, 
        WTP_income_NPV0,
        WTP_private_NPV0,
        VSLY_NPV,
        VSLY_daily_1.sum(axis = 0),
        vax_hazard.sum(axis = 0),
        VSLY_no_discount
    )

def save_metrics(metrics, name):
    np.savez_compressed(dst/f"{name}.npz", metrics)

def process(district_data):
        (state, district), state_code, N_district, N_0, N_1, N_2, N_3, N_4, N_5, N_6 = district_data
        N_jk = np.array([N_0, N_1, N_2, N_3, N_4, N_5, N_6])
        age_weight = N_jk/(N_j)
        f_hat_p1v1 = income_decline(state, district, np.zeros((simulation_range + 1, 1)), np.zeros((simulation_range + 1, 1)))
        c_p1v1 = np.transpose(
            (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[state, district].values[:, None],
            [0, 2, 1]
        )
        state_YLLs = YLLs.get(state, default = YLLs.mean(axis = 0))
        phi_cf = int(phi_points[0] * 365 * 100)
        with np.load(src/f"{state_code}_{district}_phi{phi_cf}_novax.npz") as counterfactual:
            dI_pc_p0 = counterfactual['dT']/N_district
            dD_pc_p0 = counterfactual['dD']/N_district
            q_p0v0   = counterfactual["q0"]
            D_p0     = counterfactual["Dj"]
        f_hat_p0v0 = income_decline(state, district, dI_pc_p0, dD_pc_p0)
        c_p0v0 = np.transpose(
            (1 + f_hat_p0v0) * consumption_2019.loc[state, district].values[:, None, None],
            [1, 2, 0]
        )
        YLLs_for_dist = (D_p0[-1] - D_p0[0]) @ state_YLLs
        save_metrics((D_p0[-1] - D_p0[0]).sum(axis = 1), f"evaluated_deaths_{state}_{district}_no_vax")
        save_metrics(YLLs_for_dist,  f"evaluated_YLLs_{state}_{district}_no_vax")

        wtp_nv, vsly_nv = counterfactual_metrics(q_p0v0, c_p0v0)
        save_metrics(N_jk * wtp_nv,  f"evaluated_WTP_{state}_{district}_no_vax")
        save_metrics(N_jk * vsly_nv, f"evaluated_VSLY_{state}_{district}_no_vax")
        save_metrics(D_p0, f"total_Dj_{state}_{district}_no_vax")
        save_metrics((age_weight[:, None, None] * c_p0v0.T).T, f"total_consp0v0_{state}_{district}_no_vax")
        
        for _phi in phi_points:
            phi = int(_phi * 365 * 100)
            for vax_policy in ["random", "contact", "mortality"]:
                # if (dst/f"district_YLL_{state}_{district}_{phi}_{vax_policy}.npz").exists():
                #     continue
                with np.load(src/f"{state_code}_{district}_phi{phi}_{vax_policy}.npz") as policy:
                    dI_pc_p1 = policy['dT']/N_district
                    dD_pc_p1 = policy['dD']/N_district
                    pi       = policy['pi'] 
                    q_p1v1   = policy['q1']
                    q_p1v0   = policy['q0']
                    D_p1     = policy["Dj"]

                f_hat_p1v0 = income_decline(state, district, dI_pc_p1, dD_pc_p1)
                c_p1v0 = np.transpose(
                    (1 + f_hat_p1v0) * consumption_2019.loc[state, district].values[:, None, None], 
                    [1, 2, 0]
                )

                try:
                    print(state, district, phi, vax_policy)
                    wtp, wtp_health, wtp_income, wtp_private, vsly, VSLY_daily_1, vax_hazard, VSLY_no_discount = get_metrics(pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0)
                except RuntimeWarning as rw:
                    print(rw, state, district, phi, vax_policy)
                
                save_metrics((D_p1[-1] - D_p1[0]).sum(axis = 1), f"evaluated_deaths_{state}_{district}_{phi}_{vax_policy}")
                save_metrics((D_p1[-1] - D_p1[0]) @ state_YLLs,  f"evaluated_YLL_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(VSLY_daily_1,                       f"evaluated_VSLYd1_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(vax_hazard,                         f"evaluated_vh_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(VSLY_no_discount,                   f"evaluated_VSLYND_{state}_{district}_{phi}_{vax_policy}")
                
                save_metrics(N_jk * wtp,  f"evaluated_WTP_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(N_jk * vsly, f"evaluated_VSLY_{state}_{district}_{phi}_{vax_policy}")
                
                save_metrics(age_weight * wtp_health, f"evaluated_WTP_h_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(age_weight * wtp_income, f"evaluated_WTP_i_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(age_weight * wtp_private, f"evaluated_WTP_p_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(age_weight * wtp[0], f"evaluated_WTP_pc_{state}_{district}_{phi}_{vax_policy}")

                save_metrics(D_p1, f"total_Dj_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(age_weight * c_p1v1, f"total_consp1v1_{state}_{district}_{phi}_{vax_policy}")
                save_metrics((age_weight[:, None, None] * c_p1v0.T).T, f"total_consp1v0_{state}_{district}_{phi}_{vax_policy}")
                
                save_metrics(wtp, f"district_WTP_{state}_{district}_{phi}_{vax_policy}")
                save_metrics(vsly , f"district_VSLY_{state}_{district}_{phi}_{vax_policy}")
                save_metrics((D_p1[-1] - D_p1[0]) @ state_YLLs, f"district_YLL_{state}_{district}_{phi}_{vax_policy}")


if __name__ == "__main__":
    population_columns = ["state_code", "N_tot", 'N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6']
    distribute = True

    if distribute:
        with dask.config.set({"scheduler.allowed-failures": 1}):
            client = dask.distributed.Client()#(n_workers = 1, processes = False)
            print(client.dashboard_link)
            with dask.distributed.get_task_stream(client) as ts:
                futures = []
                # for district in districts_to_run[~districts_to_run.index.isin(["Andaman And Nicobar Islands"], level = 0)][population_columns].itertuples():
                for district in districts_to_run[districts_to_run.index.isin(["Tamil Nadu"], level = 0)][population_columns].itertuples():
                    futures.append(client.submit(process, district, key = ":".join(district[0])))
            dask.distributed.progress(futures)
    else:
        for t in districts_to_run[districts_to_run.index.isin(["Tamil Nadu"], level = 0)][population_columns].itertuples():
            process(t)

