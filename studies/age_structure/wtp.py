from collections import defaultdict

import adaptive.plots as plt
import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
from studies.age_structure.palette import *
from tqdm.std import tqdm

""" Calculate willingness to pay """ 

def day_idx(df: pd.DataFrame, name = "date"):
    return df\
        .assign(**{name: pd.date_range(start = simulation_start, freq = "D", periods = len(df))})\
        .set_index(name)

def month_idx(df: pd.DataFrame, name = "month"):
    return df\
        .assign(**{name: pd.date_range(start = simulation_start, freq = "M", periods = len(df)).month})\
        .set_index(name)

# coefficients of consumption ~ prevalence regression
coeffs = pd.read_stata(data/"reg_estimates.dta")\
    [["parm", "label", "estimate"]]\
    .rename(columns = {"parm": "param", "estimate": "value"})\
    .set_index("param")

I_cat_coeffs = dict(enumerate(coeffs.filter(regex = ".*I_cat$", axis = 0).value.values))
D_cat_coeffs = dict(enumerate(coeffs.filter(regex = ".*D_cat$", axis = 0).value.values))
month_coeffs = dict(enumerate(coeffs.filter(regex = ".*month$", axis = 0).value.values, start = 1))
I_cat_natl_coeffs = coeffs\
    .filter(regex = ".*I_cat_national$", axis = 0)\
    .reset_index()\
    .assign(param = lambda df: df["param"].apply(lambda s: s.split(".")[0].replace("b", "")).astype(int))\
    .set_index("param")["value"]\
    .to_dict()

# per capita daily consumption levels 
consumption_2019 = pd.read_stata("data/pcons_2019m6.dta")\
    .set_index("districtnum")\
    .rename(index = {"Kanniyakumari": "Kanyakumari"})

# national-level forward simulation
IN_simulated_percap = pd.read_csv("data/IN_simulated_percap.csv")\
    .assign(month = lambda _: _.month.str.zfill(7))\
    .set_index("month")\
    .sort_index()

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

I_cat_national_mapping = {
    1: 10, 
    2: 7,
    3: 4,
    4: 3,
    5: 3,
    6: 2,
    7: 2,
    8: 1, 
    9: 1,
    10: 1,
    11: 1,
    12: 0
}

def cut(column, cutoffs):
    return lambda _: (1 + pd.cut(_[column], [0] + cutoffs + [1], labels = False)).fillna(0).astype(int)

# national prevalence categories are sparse and easily mapped as a function of time
def date_to_I_cat_natl(date: pd.Timestamp):
    if date.year != 2021:
        return 0
    return I_cat_national_mapping[date.month]

def coeff_label(suffix, dropped):
    return lambda i: str(i) + ("b" if i == dropped else "") + "." + suffix

def coeff_value(column, dropped):
    return lambda _: coeffs.loc[_[column].apply(lambda i: str(i) + ("b" if i == dropped else "") + "." + column)].value.values

def estimate_consumption_decline(district, dI_pc, dD_pc, force_natl_zero = False):
    district_code = str(district_codes[district])
    district_coeff_label = district_code + ("b" if district_code == "92" else "") + ".districtnum"
    constant = coeffs.loc["_cons"].value + coeffs.loc[district_coeff_label].value

    # map values to categoricals 
    indicators = pd.DataFrame({
        "I_cat": np.nan_to_num(1 + pd.cut(dI_pc, infection_cutoffs, labels = False)).astype(int),
        "D_cat": np.nan_to_num(1 + pd.cut(dD_pc,     death_cutoffs, labels = False)).astype(int),
        "date": pd.date_range(start = simulation_start, freq = "D", periods = len(dI_pc))
    }).assign(
        month          = lambda _: _.date.dt.month,
        I_cat_national = lambda _: _.date.apply(date_to_I_cat_natl if not force_natl_zero else lambda _: 0)
    )

    # index categoricals into coefficients and sum
    return indicators.assign(
        I_coeff     = coeff_value("I_cat",          dropped = 0),
        D_coeff     = coeff_value("D_cat",          dropped = 0),
        I_nat_coeff = coeff_value("I_cat_national", dropped = 0),
        month_coeff = coeff_value("month",          dropped = 1),
    )\
    .set_index("date")\
    .filter(like = "coeff", axis = 1)\
    .sum(axis = 1) + constant

def dict_map(arr, mapping): 
    # from https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    mapping_ar = np.zeros(k.max()+1, dtype = v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[arr]

def fhat(district, dI_pc, dD_pc, force_natl_zero = False):
    district_code = str(district_codes[district])
    district_coeff_label = district_code + ("b" if district_code == "92" else "") + ".districtnum"
    constant = coeffs.loc["_cons"].value + coeffs.loc[district_coeff_label].value

    I_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_, infection_cutoffs, labels = False), 0, dI_pc)).astype(int)
    D_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_,     death_cutoffs, labels = False), 0, dD_pc)).astype(int)
    month = pd.date_range(start = simulation_start, periods = len(dI_pc), freq = "D").month.values
    I_cat_national = np.where(np.arange(len(dI_pc)) >= 365, 0, dict_map(month, I_cat_national_mapping))

    I_cat_coeff_values = dict_map(I_cat, I_cat_coeffs)
    D_cat_coeff_values = dict_map(D_cat, D_cat_coeffs)
    month_coeff_values = dict_map(month, month_coeffs)
    I_cat_national_coeff_values = dict_map(I_cat_national, I_cat_natl_coeffs)

def daily_WTP_old(district, N_district, dI_pc, dD_pc, Dx_v, Dx_nv, ve = 0.7):
    f_hat_nv = estimate_consumption_decline(district, dI_pc, dD_pc)
    consumption_nv = (1 + f_hat_nv)[:, None] * consumption_2019.loc[district].values

    P_death_nv = Dx_nv.diff().shift(-1).fillna(0).cumsum()/split_by_age(N_district) 
    WTP_nv = ((1 - P_death_nv) * consumption_nv).set_index(f_hat_nv.index)

    # vaccination 
    f_hat_v = estimate_consumption_decline(district, pd.Series([0]*len(dI_pc)), pd.Series([0]*len(dI_pc)))
    consumption_v = (1 + f_hat_v)[:, None] * consumption_2019.loc[district].values

    P_death_v = Dx_v.diff().shift(-1).fillna(0).cumsum()/split_by_age(N_district) 
    WTP_v = ((ve + (1-ve)*(1 - P_death_v)) * consumption_v)\
        .set_index(pd.date_range(start = simulation_start,  end = simulation_start + pd.Timedelta(len(Dx_v) - 1, "days")))

    # where vaccination time series is longer than no vax timeseries, we assume consumption is identical 
    return (WTP_v - WTP_nv).fillna(0) 

def daily_WTP_health(district, N_district, dI_pc_v, dD_pc_v, Dx_v, Dx_nv, ve = 0.7):
    P_death_v  = Dx_v.diff().shift(-1).fillna(0).cumsum()/split_by_age(N_district) 
    q_v = 1 - P_death_v

    P_death_nv = Dx_nv.diff().shift(-1).fillna(0).cumsum()/split_by_age(N_district) 
    q_nv = 1 - P_death_nv 

    v_consumption_v  = (1 + estimate_consumption_decline(district, pd.Series([0] * (1 + 5 * 365)), pd.Series([0] * (1 + 5 * 365))) )[:, None] * consumption_2019.loc[district].values
    v_consumption_nv = (1 + estimate_consumption_decline(district, dI_pc_v.reindex(range(1826), fill_value = 0),  dD_pc_v.reindex(range(1826), fill_value = 0)))[:, None] * consumption_2019.loc[district].values

    P_vax = (0.5/365 * np.arange(5 * 365 + 1)).clip(0, 1)[:, None]

    return (1 - P_vax) * (q_v - q_nv) * v_consumption_nv +  P_vax * (1 - q_nv) * v_consumption_v

def daily_WTP_income(district, N_district, dI_pc_nv, dD_pc_nv, dI_pc_v, dD_pc_v, Dx_nv, ve = 0.7):
    P_death_nv = Dx_nv.diff().shift(-1).fillna(0).cumsum()/split_by_age(N_district) 
    q_nv = 1 - P_death_nv 

    v_consumption_v  = (1 + estimate_consumption_decline(district, pd.Series([0] * (1 + 5 * 365)), pd.Series([0] * (1 + 5 * 365))))[:, None] * consumption_2019.loc[district].values
    v_consumption_nv = (1 + estimate_consumption_decline(district, dI_pc_v.reindex(range(1826), fill_value = 0),  dD_pc_v.reindex(range(1826), fill_value = 0)) )[:, None] * consumption_2019.loc[district].values
    nv_consumption   = (1 + estimate_consumption_decline(district, dI_pc_nv, dD_pc_nv) )[:, None] * consumption_2019.loc[district].values

    P_vax = (0.5/365 * np.arange(5 * 365 + 1)).clip(0, 1)[:, None]

    return (1 - P_vax) * q_nv * (v_consumption_nv - nv_consumption) +  P_vax * q_nv * (v_consumption_v - nv_consumption)

# satej's method
def discounted_WTP(wtp, rate = (4.25/100), period = "daily"):
    if period == "daily":
        rate /= 365
    elif period == "monthly":
        rate /= 12 
    return (wtp * ((1 + rate) ** -np.arange(len(wtp)))[:, None]).sum(axis = 0)

def latex_table_row(rowname, items):
    return " & ".join([rowname] + list(items.values.round(2).astype(str))) + " \\\\ "

def get_metrics(
    pi, 
    q_p1v1, q_p1v0, q_p0v0, 
    c_p1v1, c_p1v0, c_p0v0 
):  
    dWTP_daily = \
        (1 - pi) * q_p1v0 * c_p1v0.T +\
             pi  * q_p1v1 * c_p1v1[:, None] - q_p0v0 * c_p0v0.T 

    dWTP_health_daily = \
        (1 - pi) * (q_p1v0 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v0.T +\
             pi  * (q_p1v1 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v1[:, None]

    dWTP_income_daily = \
        (1 - pi) * q_p0v0.mean(axis = 1)[:, None, :] * (c_p1v0 - c_p0v0.mean(axis = 1)[:, None, :]).T +\
             pi  * q_p1v1.mean(axis = 1)[:, None, :] * (c_p1v1 - c_p0v0.mean(axis = 1).T)[:, None, :]

    dWTP_private_daily = \
        q_p1v1 * c_p1v1[:, None] - q_p1v0 * c_p1v0.T

    VSLY_daily_1 = ((1 - pi) * q_p1v0 + pi * q_p1v1) * np.mean(c_p1v0, axis = 1).T[:, None, :]
    VSLY_daily_0 = q_p0v0 * np.mean(c_p0v0, axis = 1).T[:, None, :]
    dVLSY_daily = VSLY_daily_1 - VSLY_daily_0

    beta = 1/(1 + 4.25/365)
    s = np.arange(simulation_range + 1)

    WTP      = [] 
    VSLY     = []
    WTP_private = None
    WTP_health = None
    WTP_income = None
    for t in range(simulation_range + 1):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * dWTP_daily[t:, :], axis = 0)
        WTP.append(wtp)

        vsly = np.sum(np.power(beta, s[t:])[:, None, None] * dVLSY_daily[t:, :], axis = 0)
        VSLY.append(vsly)

        if t == 0:
            WTP_health  = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_health_daily [t:, :], axis = 0)
            WTP_income  = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_income_daily [t:, :], axis = 0)
            WTP_private = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_private_daily[t:, :], axis = 0) 

    return (
        WTP,
        WTP_health, 
        WTP_income,
        WTP_private,
        VSLY,
    )

if __name__ == "__main__":
    evaluated_deaths = defaultdict(lambda: np.zeros(num_sims))
    evaluated_YLL    = defaultdict(lambda: np.zeros(num_sims))
    evaluated_WTP    = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    evaluated_WTP_h  = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    evaluated_WTP_i  = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    evaluated_WTP_p  = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    evaluated_VSLY   = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    district_WTP     = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))

    progress = tqdm(total = len(districts_to_run) * 4)
    for (district, _, N_district, *_) in districts_to_run.itertuples():
        progress.set_description(f"{district:15s}|    no vax|         ")
        f_hat_p1v1 = estimate_consumption_decline(district, 
            pd.Series(np.zeros(simulation_range + 1)), 
            pd.Series(np.zeros(simulation_range + 1)), force_natl_zero = True)
        c_p1v1 = (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values

        with np.load(data/f"sim_metrics/{state}_{district}_phi25_novax.npz") as counterfactual:
            dI_pc_p0 = counterfactual['dT']/N_district
            dD_pc_p0 = counterfactual['dD']/N_district
            q_p0v0   = counterfactual["q0"]
            D_p0     = counterfactual["Dj"]
        f_hat_p0v0 = np.array([estimate_consumption_decline(district, dI_pc_p0[:, _], dD_pc_p0[:, _]) for _ in range(num_sims)])
        c_p0v0 = (1 + f_hat_p0v0) * consumption_2019.loc[district].values[:, None, None]

        evaluated_deaths[25, "no_vax"] += (D_p0[-1] - D_p0[0]).sum(axis = 1)
        evaluated_YLL   [25, "no_vax"] += (D_p0[-1] - D_p0[0]) @ YLLs

        progress.update(1)
        
        for _phi in phi_points:
            phi = int(_phi * 365 * 100)

            for vax_policy in ["random", "contact", "mortality"]:
                progress.set_description(f"{district:15s}| {vax_policy:>9s}| φ = {str(int(phi)):>3s}%")
                with np.load(data/f"sim_metrics/{state}_{district}_phi{phi}_{vax_policy}.npz") as policy:
                    dI_pc_p1 = policy['dT']/N_district
                    dD_pc_p1 = policy['dD']/N_district
                    pi       = policy['pi'] 
                    q_p1v1   = policy['q1']
                    q_p1v0   = policy['q0']
                    D_p1     = policy["Dj"]

                f_hat_p1v0 = np.array([estimate_consumption_decline(district, dI_pc_p1[:, _], dD_pc_p1[:, _]) for _ in range(num_sims)])
                c_p1v0 = (1 + f_hat_p1v0) * consumption_2019.loc[district].values[:, None, None]

                wtp, wtp_health, wtp_income, wtp_private, vsly = get_metrics(pi, q_p1v1, q_p1v0, q_p0v0, c_p1v1, c_p1v0, c_p0v0)
                
                evaluated_deaths[phi, vax_policy] += (D_p1[-1] - D_p1[0]).sum(axis = 1)
                evaluated_YLL   [phi, vax_policy] += (D_p1[-1] - D_p1[0]) @ YLLs
                evaluated_WTP   [phi, vax_policy] += wtp
                evaluated_WTP_h [phi, vax_policy] += wtp_health
                evaluated_WTP_i [phi, vax_policy] += wtp_income
                evaluated_WTP_p [phi, vax_policy] += wtp_private
                evaluated_VSLY  [phi, vax_policy] += vsly
                if phi == 50 and vax_policy == "random":
                    district_WTP[district] = wtp
                progress.update(1)

    death_percentiles = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_deaths.items()}
    YLL_percentiles   = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_YLL.items()}
    VSLY_percentiles  = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_VSLY.items()}
    WTP_percentiles   = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_WTP.items()}

    # death outcomes 
    #region
    fig = plt.gcf()

    md, lo, hi = death_percentiles[(25, "no_vax")]
    *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(25, "random")]
    *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(25, "contact")]
    *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(25, "mortality")]
    *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(50, "random")]
    *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(50, "contact")]
    *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(50, "mortality")]
    *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(100, "random")]
    *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(100, "contact")]
    *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(100, "mortality")]
    *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks([0, 1, 2, 3], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.PlotDevice().ylabel("deaths\n")
    plt.show()
    # #endregion

    # # YLL 
    # #region
    fig = plt.figure()

    md, lo, hi = YLL_percentiles[(25, "no_vax")]
    *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(25, "random")]
    *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(25, "contact")]
    *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(25, "mortality")]
    *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(50, "random")]
    *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(50, "contact")]
    *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(50, "mortality")]
    *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]


    md, lo, hi = YLL_percentiles[(100, "random")]
    *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(100, "contact")]
    *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = YLL_percentiles[(100, "mortality")]
    *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks([0, 1, 2, 3], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.PlotDevice().ylabel("YLLs\n")
    plt.show()
    #endregion

    # WTP
    #region
    fig = plt.figure()

    # md, lo, hi = WTP_percentiles[(25, "no_vax")]
    # *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    #     fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
    # [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(25, "random")] * USD
    *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(25, "contact")] * USD
    *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(25, "mortality")] * USD
    *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(50, "random")] * USD
    *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(50, "contact")] * USD
    *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(50, "mortality")] * USD
    *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(100, "random")] * USD
    *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(100, "contact")] * USD
    *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = WTP_percentiles[(100, "mortality")] * USD
    *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks([1, 2, 3], ["$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.show()
    #endregion

    # VSLY 
    #region
    fig = plt.figure()

    # md, lo, hi = WTP_percentiles[(25, "no_vax")]
    # *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    #     fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
    # [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(25, "random")] * USD
    *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(25, "contact")] * USD
    *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(25, "mortality")] * USD
    *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(50, "random")] * USD
    *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(50, "contact")] * USD
    *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(50, "mortality")] * USD
    *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(100, "random")] * USD
    *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(100, "contact")] * USD
    *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = VSLY_percentiles[(100, "mortality")] * USD
    *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks([1, 2, 3], ["$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.PlotDevice().ylabel("VSLY (USD)\n")
    plt.show()
    #endregion

    # aggregate WTP by age
    WTP_random_50_percentile = np.percentile(evaluated_WTP[0.5, "random"][0, :, :], [50, 5, 95], axis = 0) * USD
    fig = plt.figure()
    for (i, (md, lo, hi)) in enumerate(WTP_random_50_percentile.T):
        *_, bars = plt.errorbar(x = [i], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
        [_.set_alpha(0.5) for _ in bars]
    plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.legend(title = "age bin", title_fontsize = "25", fontsize = "25")
    plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
    plt.show()

    print("")

    # health/consumption
    summed_wtp_health = np.median(np.sum([np.array(_) for _ in evaluated_WTP_h.values()], axis = 0), axis = 0)
    summed_wtp_income = np.median(np.sum([np.array(_) for _ in evaluated_WTP_i.values()], axis = 0), axis = 0)
    fig, ax = plt.subplots()
    ax.bar(range(7), summed_wtp_income * USD, bottom = summed_wtp_health * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2)
    ax.bar(range(7), summed_wtp_health * USD,                                   color = age_group_colors, edgecolor = age_group_colors, linewidth = 2)
    ax.bar(range(7), [0], label = "income", color = "white", edgecolor = "black", linewidth = 2)
    ax.bar(range(7), [0], label = "health", color = "black", edgecolor = "black", linewidth = 2)
    plt.xticks(range(7), age_bin_labels, fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.semilogy()
    plt.show()

    # social/private 
    summed_wtp_priv = np.median(np.sum([np.array(_) for _ in evaluated_WTP_p.values()], axis = 0), axis = 0)
    summed_wtp_tot  = np.median(np.sum([np.array(_) for _ in district_WTP.values()], axis = 0)[0, :, :], axis = 0)
    # summed_wtp      = np.median(np.sum([np.array(_) for _ in evaluated_WTP  .values()], axis = 0)[0, :, :], axis = 0)
    summed_wtp_soc  = summed_wtp_tot - summed_wtp_priv
    fig, ax = plt.subplots()
    ax.bar(range(7), summed_wtp_priv * USD, bottom = summed_wtp_soc * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2)
    ax.bar(range(7), summed_wtp_soc  * USD,                                color = age_group_colors, edgecolor = age_group_colors, linewidth = 2)
    ax.bar(range(7), [0], label = "social",  color = "white", edgecolor = "black", linewidth = 2)
    ax.bar(range(7), [0], label = "private", color = "black", edgecolor = "black", linewidth = 2)

    plt.xticks(range(7), age_bin_labels, fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.semilogy()
    plt.show()

    # dist x age 
    per_district_percentiles = {district: np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for (district, wtp) in per_district_WTPs.items()}

    fig = plt.figure()
    district_ordering = list(per_district_percentiles.keys())[:5]
    for (i, district) in enumerate(district_ordering):
        wtps = per_district_percentiles[district]
        for j in range(7):
            plt.errorbar(
                x = [i + 0.1 * (j - 3)],
                y = wtps[0, 6-j] * USD,
                yerr = [
                    [(wtps[0, 6-j] - wtps[1, 6-j]) * USD],
                    [(wtps[2, 6-j] - wtps[0, 6-j]) * USD]
                ], 
                fmt = "o",
                color = age_group_colors[6-j],
                figure = fig,
                label = None if i > 0 else age_bin_labels[6-j]
            )
    plt.xticks(
        range(len(district_ordering)),
        district_ordering,
        rotation = 45,
        fontsize = "25"
    )
    plt.yticks(fontsize = "25")
    plt.legend(title = "age bin", title_fontsize = "25", fontsize = "25")
    # plt.ylim(0, 10000)
    plt.xlim(-0.5, len(district_ordering) - 0.5)
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.show()
