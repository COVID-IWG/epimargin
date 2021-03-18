from collections import defaultdict

import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
from tqdm.std import tqdm

""" Calculate WTP, VSLY, and other policy evaluation metrics """ 

src = mkdir(data/f"sim_metrics{num_sims}")
dst = mkdir(data/f"wtp_metrics{num_sims}")

# coefficients of consumption ~ prevalence regression
coeffs = pd.read_stata(data/"reg_estimates_full.dta")\
    [["parm", "label", "estimate"]]\
    .rename(columns = {"parm": "param"})\
    .set_index("param")

I_cat_coeffs = dict(enumerate(coeffs.filter(regex = ".*I_cat$", axis = 0).estimate.values))
D_cat_coeffs = dict(enumerate(coeffs.filter(regex = ".*D_cat$", axis = 0).estimate.values))
month_coeffs = dict(enumerate(coeffs.filter(regex = ".*month$", axis = 0).estimate.values, start = 1))
district_coeffs = coeffs.filter(regex = ".*districtnum$", axis = 0)\
    .estimate.reset_index()\
    .assign(param = lambda df: df["param"].str[:3].str.replace("b", "").astype(int))\
    .set_index("param").to_dict()["estimate"]

# per capita daily consumption levels 
consumption_2019 = pd.read_stata("data/pcons_2019m6.dta")\
    .set_index("districtnum")\
    .rename(index = {"Kanniyakumari": "Kanyakumari"})

# bin cutoffs for prevalence categories
infection_cutoffs = [0, 
    1.31359918148e-06,
    2.78728440853e-06,
    5.98520919658e-06,
    9.01749487694e-06,
    .0000138806432497,
    .0000232067541053,
    .0000348692029503,
    .0000553322569194,
    .0000837807402432, 
    1
]

death_cutoffs = [0, 
    4.86518215364e-08,
    7.74357252506e-08,
    1.17110323077e-07,
    1.79850716711e-07,
    3.08246710742e-07,
    4.38650091191e-07,
    6.63577948309e-07,
    9.89375901681e-07,
    1.52713164555e-06,
    1
]

def dict_map(arr, mapping): 
    # from https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    mapping_ar = np.zeros(k.max()+1, dtype = v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[arr]

def income_decline(district_code, dI_pc, dD_pc):
    constant = coeffs.loc["_cons"].estimate + district_coeffs[district_code]

    I_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_, infection_cutoffs, labels = False), 0, dI_pc)).astype(int)
    D_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_,     death_cutoffs, labels = False), 0, dD_pc)).astype(int)
    month = pd.date_range(start = simulation_start, periods = len(dI_pc), freq = "D").month.values

    return (
        dict_map(I_cat, I_cat_coeffs)          + 
        dict_map(D_cat, D_cat_coeffs)          + 
        dict_map(month, month_coeffs)[:, None] + 
        constant
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
    
    beta = 1/(1 + 4.25/365)
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

    VSLY_daily_1 = ((1 - pi) * q_p1v0 + pi * q_p1v1) * np.mean(c_p0v0, axis = 1)[:, None, :]

    cb_daily_1 = (WTP_daily_1 - VSLY_daily_1)

    beta = 1/(1 + 4.25/365)
    s = np.arange(simulation_range + 1)

    WTP_NPV  = [] 
    VSLY_NPV = []
    WTP_health_NPV0  = None
    WTP_income_NPV0  = None
    WTP_private_NPV0 = None
    for t in range(simulation_range + 1):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * WTP_daily_1[t:, :], axis = 0)
        WTP_NPV.append(wtp)

        vsly = np.sum(np.power(beta, s[t:] - t)[:, None, None] * VSLY_daily_1[t:, :], axis = 0)
        VSLY_NPV.append(vsly)

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
    )

def save_metrics(metrics, name):
    np.savez(dst/f"{name}.npz", **{"_".join(map(str, k)): v for (k, v) in metrics.items()})


if __name__ == "__main__":
    evaluated_WTP    = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    evaluated_VSLY   = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    total_Dj         = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    total_consp1v0   = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    total_consp0v0   = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    total_consp1v1   = defaultdict(lambda: np.zeros((simulation_range + 1, 1, num_age_bins)))
    evaluated_WTP_h  = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    evaluated_WTP_i  = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    evaluated_WTP_p  = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    evaluated_WTP_s  = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    evaluated_WTP_pc = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    district_WTP     = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    district_YLL     = defaultdict(lambda: np.zeros((num_sims, num_age_bins)))
    evaluated_deaths = defaultdict(lambda: np.zeros(num_sims))
    evaluated_YLL    = defaultdict(lambda: np.zeros(num_sims))

    progress = tqdm(total = len(districts_to_run) * (1 + len(phi_points) * 3) + 13)
    for (district, N_district, N_0, N_1, N_2, N_3, N_4, N_5, N_6) in districts_to_run[["N_tot", 'N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6']].itertuples():
        progress.set_description(f"{district:15s}|    no vax|         ")
        district_code = district_codes[district]
        N_jk = np.array([N_0, N_1, N_2, N_3, N_4, N_5, N_6])
        age_weight = N_jk/(N_j)
        pop_weight = N_jk/(N_j.sum())
        f_hat_p1v1 = income_decline(district_code, np.zeros((simulation_range + 1, 1)), np.zeros((simulation_range + 1, 1)))
        c_p1v1 = np.transpose(
            (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values[:, None],
            [0, 2, 1]
        )

        with np.load(src/f"{state}_{district}_phi25_novax.npz") as counterfactual:
            dI_pc_p0 = counterfactual['dT']/N_district
            dD_pc_p0 = counterfactual['dD']/N_district
            q_p0v0   = counterfactual["q0"]
            D_p0     = counterfactual["Dj"]
        f_hat_p0v0 = income_decline(district_code, dI_pc_p0, dD_pc_p0)
        c_p0v0 = np.transpose(
            (1 + f_hat_p0v0) * consumption_2019.loc[district].values[:, None, None],
            [1, 2, 0]
        )

        evaluated_deaths[25, "no_vax"] += (D_p0[-1] - D_p0[0]).sum(axis = 1)
        evaluated_YLL   [25, "no_vax"] += (D_p0[-1] - D_p0[0]) @ YLLs

        wtp_nv, vsly_nv = counterfactual_metrics(q_p0v0, c_p0v0)
        evaluated_WTP   [25, "no_vax"] += N_jk * wtp_nv
        evaluated_VSLY  [25, "no_vax"] += N_jk * vsly_nv
        total_Dj        [25, "no_vax"] += D_p0 
        total_consp0v0  [25, "no_vax"] += (age_weight[:, None, None] * c_p0v0.T).T

        progress.update(1)
        
        for _phi in phi_points:
            phi = int(_phi * 365 * 100)

            for vax_policy in ["random", "contact", "mortality"]:
                progress.set_description(f"{district:15s}| {vax_policy:>9s}| φ = {str(int(phi)):>3s}%")
                with np.load(src/f"{state}_{district}_phi{phi}_{vax_policy}.npz") as policy:
                    dI_pc_p1 = policy['dT']/N_district
                    dD_pc_p1 = policy['dD']/N_district
                    pi       = policy['pi'] 
                    q_p1v1   = policy['q1']
                    q_p1v0   = policy['q0']
                    D_p1     = policy["Dj"]

                f_hat_p1v0 = income_decline(district_code, dI_pc_p0, dD_pc_p0)
                c_p1v0 = np.transpose(
                    (1 + f_hat_p1v0) * consumption_2019.loc[district].values[:, None, None], 
                    [1, 2, 0]
                )

                wtp, wtp_health, wtp_income, wtp_private, vsly = get_metrics(pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0)
                
                evaluated_deaths[phi, vax_policy] += (D_p1[-1] - D_p1[0]).sum(axis = 1)
                evaluated_YLL   [phi, vax_policy] += (D_p1[-1] - D_p1[0]) @ YLLs.values
                
                evaluated_WTP   [phi, vax_policy] += N_jk * wtp
                evaluated_VSLY  [phi, vax_policy] += N_jk * vsly
                
                evaluated_WTP_h  [phi, vax_policy] += age_weight * wtp_health
                evaluated_WTP_i  [phi, vax_policy] += age_weight * wtp_income
                evaluated_WTP_p  [phi, vax_policy] += age_weight * wtp_private
                # evaluated_WTP_s  [phi, vax_policy] += age_weight * wtp_social
                evaluated_WTP_pc [phi, vax_policy] += age_weight * wtp[0]

                total_Dj[phi, vax_policy] += D_p1
                total_consp1v1[phi, vax_policy] += age_weight * c_p1v1
                total_consp1v0[phi, vax_policy] += (age_weight[:, None, None] * c_p1v0.T).T
                
                district_WTP[district, phi, vax_policy] = N_jk * wtp
                district_YLL[district, phi, vax_policy] = (D_p1[-1] - D_p1[0]) * YLLs.values
                progress.update(1)

    metric_names = {
        "evaluated_deaths": evaluated_deaths,
        "evaluated_YLL"   : evaluated_YLL,
        "evaluated_WTP"   : evaluated_WTP,
        "evaluated_VSLY"  : evaluated_VSLY,
        "evaluated_WTP_h" : evaluated_WTP_h,
        "evaluated_WTP_i" : evaluated_WTP_i,
        "evaluated_WTP_p" : evaluated_WTP_p,
        "evaluated_WTP_pc": evaluated_WTP_pc,
        "total_Dj"        : total_Dj,
        "total_consp1v1"  : total_consp1v1,
        "total_consp1v0"  : total_consp1v0,
        "district_WTP"    : district_WTP,
        "district_YLL"    : district_YLL,
    }
    progress.set_description("serializing datasets                ")
    for (name, metrics) in metric_names.items():
        save_metrics(metrics, name)
        progress.update(1)
