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

# national prevalence categories are sparse and easily mapped as a function of time
def date_to_I_cat_natl(date: pd.Timestamp):
    if date.year != 2021:
        return 0
    return I_cat_national_mapping[date.month]

def dict_map(arr, mapping): 
    # from https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    mapping_ar = np.zeros(k.max()+1, dtype = v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[arr]

def income_decline(district, dI_pc, dD_pc, force_natl_zero = False):
    district_code = str(district_codes[district])
    district_coeff_label = district_code + ("b" if district_code == "92" else "") + ".districtnum"
    constant = coeffs.loc["_cons"].value + coeffs.loc[district_coeff_label].value

    I_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_, infection_cutoffs, labels = False), 0, dI_pc)).astype(int)
    D_cat = np.nan_to_num(np.apply_along_axis(lambda _: 1 + pd.cut(_,     death_cutoffs, labels = False), 0, dD_pc)).astype(int)
    month = pd.date_range(start = simulation_start, periods = len(dI_pc), freq = "D").month.values
    if force_natl_zero: 
        I_cat_national = np.zeros(month.shape).astype(int)
    else:
        I_cat_national = np.where(np.arange(len(dI_pc)) >= 365, 0, dict_map(month, I_cat_national_mapping))

    I_cat_coeff_values = dict_map(I_cat, I_cat_coeffs)
    D_cat_coeff_values = dict_map(D_cat, D_cat_coeffs)
    month_coeff_values = dict_map(month, month_coeffs)
    I_cat_national_coeff_values = dict_map(I_cat_national, I_cat_natl_coeffs)

    return ((I_cat_coeff_values + D_cat_coeff_values) + (month_coeff_values + I_cat_national_coeff_values)[:, None] + constant).T

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
             pi  * q_p1v1 * c_p1v1 - q_p0v0 * c_p0v0.T 

    dWTP_health_daily = \
        (1 - pi) * (q_p1v0 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v0.T +\
             pi  * (q_p1v1 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v1

    dWTP_income_daily = \
        (1 - pi) * q_p0v0.mean(axis = 1)[:, None, :] * (c_p1v0 - c_p0v0.mean(axis = 1)[:, None, :]).T +\
             pi  * q_p1v1.mean(axis = 1)[:, None, :] * (c_p1v1 - c_p0v0.mean(axis = 1)[:, None, :].T)

    dWTP_private_daily = \
        q_p1v1 * c_p1v1 - q_p1v0 * c_p1v0.T

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
    VSLY_0           = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))
    district_WTP     = defaultdict(lambda: np.zeros((simulation_range + 1, num_sims, num_age_bins)))

    progress = tqdm(total = len(districts_to_run) * 12)
    for (district, _, N_district, *_) in districts_to_run.itertuples():
        progress.set_description(f"{district:15s}|    no vax|         ")
        f_hat_p1v1 = income_decline(district, np.zeros((simulation_range + 1, 1)), np.zeros((simulation_range + 1, 1)), force_natl_zero = True)
        c_p1v1 = np.transpose(
            (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values[:, None],
            [2, 0, 1]
        )

        with np.load(data/f"sim_metrics_100/{state}_{district}_phi25_novax.npz") as counterfactual:
            dI_pc_p0 = counterfactual['dT']/N_district
            dD_pc_p0 = counterfactual['dD']/N_district
            q_p0v0   = counterfactual["q0"]
            D_p0     = counterfactual["Dj"]
        f_hat_p0v0 = income_decline(district, dI_pc_p0, dD_pc_p0)
        c_p0v0 = (1 + f_hat_p0v0) * consumption_2019.loc[district].values[:, None, None]

        evaluated_deaths[25, "no_vax"] += (D_p0[-1] - D_p0[0]).sum(axis = 1)
        evaluated_YLL   [25, "no_vax"] += (D_p0[-1] - D_p0[0]) @ YLLs

        progress.update(1)
        
        for _phi in phi_points:
            phi = int(_phi * 365 * 100)

            for vax_policy in ["random", "contact", "mortality"]:
                progress.set_description(f"{district:15s}| {vax_policy:>9s}| φ = {str(int(phi)):>3s}%")
                with np.load(data/f"sim_metrics_100/{state}_{district}_phi{phi}_{vax_policy}.npz") as policy:
                    dI_pc_p1 = policy['dT']/N_district
                    dD_pc_p1 = policy['dD']/N_district
                    pi       = policy['pi'] 
                    q_p1v1   = policy['q1']
                    q_p1v0   = policy['q0']
                    D_p1     = policy["Dj"]

                f_hat_p1v0 = income_decline(district, dI_pc_p0, dD_pc_p0)
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


    # death outcomes - min/mean/max
    #region
    fig = plt.gcf()

    md, lo, hi = death_percentiles[(25, "no_vax")]
    md, lo, hi = [op(evaluated_deaths[25, "no_vax"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(25, "random")]
    md, lo, hi = [op(evaluated_deaths[25, "random"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(25, "contact")]
    md, lo, hi = [op(evaluated_deaths[25, "contact"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(25, "mortality")]
    md, lo, hi = [op(evaluated_deaths[25, "mortality"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(50, "random")]
    md, lo, hi = [op(evaluated_deaths[50, "random"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(50, "contact")]
    md, lo, hi = [op(evaluated_deaths[50, "contact"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(50, "mortality")]
    md, lo, hi = [op(evaluated_deaths[50, "mortality"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(100, "random")]
    md, lo, hi = [op(evaluated_deaths[100, "random"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(100, "contact")]
    md, lo, hi = [op(evaluated_deaths[100, "contact"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(100, "mortality")]
    md, lo, hi = [op(evaluated_deaths[100, "mortality"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    md, lo, hi = death_percentiles[(200, "random")]
    md, lo, hi = [op(evaluated_deaths[200, "random"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [4 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(200, "contact")]
    md, lo, hi = [op(evaluated_deaths[200, "contact"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [4], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    md, lo, hi = death_percentiles[(200, "mortality")]
    md, lo, hi = [op(evaluated_deaths[200, "mortality"]) for op in (np.mean, np.min, np.max)]
    *_, bars = plt.errorbar(x = [4 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]

    ##################

    # md, lo, hi = death_percentiles[(500, "random")]
    # md, lo, hi = [op(evaluated_deaths[500, "random"]) for op in (np.mean, np.min, np.max)]
    # *_, bars = plt.errorbar(x = [5 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    #     fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
    # [_.set_alpha(0.5) for _ in bars]

    # md, lo, hi = death_percentiles[(500, "contact")]
    # md, lo, hi = [op(evaluated_deaths[500, "contact"]) for op in (np.mean, np.min, np.max)]
    # *_, bars = plt.errorbar(x = [5], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    #     fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
    # [_.set_alpha(0.5) for _ in bars]

    # md, lo, hi = death_percentiles[(500, "mortality")]
    # md, lo, hi = [op(evaluated_deaths[500, "mortality"]) for op in (np.mean, np.min, np.max)]
    # *_, bars = plt.errorbar(x = [5 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    #     fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
    # [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "25", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks([0, 1, 2, 3, 4], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%", "$\phi = 200$%"], fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.PlotDevice().ylabel("deaths\n")
    plt.show()
    # #endregion

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

    # md, lo, hi = np.percentile(np.sum(list(VSLY_0.values())), [50, 5, 95], axis = 1)
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
    # WTP_random_50_percentile = np.percentile(evaluated_WTP[50, "random"][0, :, :], [50, 5, 95], axis = 0) * USD
    fig = plt.figure()
    for (i, (md, lo, hi)) in enumerate(zip(*np.percentile(np.sum([v[0] for v in district_WTP.values()], axis = 0), [50, 5, 95], axis = 0))):
        *_, bars = plt.errorbar(x = [i], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
        [_.set_alpha(0.5) for _ in bars]
    plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.legend(title = "age bin", title_fontsize = "25", fontsize = "25")
    plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
    plt.semilogy()
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
    per_district_percentiles = {district: np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for (district, wtp) in district_WTP.items()}

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
