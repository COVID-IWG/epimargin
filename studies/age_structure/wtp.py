import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from collections import defaultdict
import adaptive.plots as plt

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
    .set_index("parm")\
    .rename(columns = {"parm": "param", "estimate": "value"})

# per capita daily consumption levels 
consumption_2019 = pd.read_stata("data/pcons_2019m6.dta")\
    .set_index("districtnum")\
    .rename(index = {"Kanniyakumari": "Kanyakumari"})

# national-level forward simulation
IN_simulated_percap = pd.read_csv("data/IN_simulated_percap.csv")\
    .assign(month = lambda _: _.month.str.zfill(7))\
    .set_index("month")\
    .sort_index()

N_Chennai = np.array([1206512, 1065145, 822092, 646810, 442051, 273618, 183610])

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

def cut(column, cutoffs):
    return lambda _: (1 + pd.cut(_[column], [0] + cutoffs + [1], labels = False)).fillna(0).astype(int)

# national prevalence categories are sparse and easily mapped as a function of time
def date_to_I_cat_natl(date: pd.Timestamp):
    if date.year != 2021:
        return 0
    return {
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
    }[date.month]

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

# alice's method
def avg_monthly_WTP(wtp):
    return 30 * wtp.groupby(wtp.index.month.astype(str).str.zfill(2) + "_" + wtp.index.year.astype(str)).mean().mean()

def latex_table_row(rowname, items):
    return " & ".join([rowname] + list(items.values.round(2).astype(str))) + " \\\\ "

if __name__ == "__main__":
    # run calculations and then print out each aggregation
    # daily_WTP_calcs = {district: daily_WTP(district) for district in district_codes.keys()}
    wtp_daily  = defaultdict(lambda: 0)
    wtp_total  = defaultdict(lambda: 0)
    wtp_range  = defaultdict(lambda: 0)
    wtp_percentiles  = defaultdict(lambda: 0)
    wtp_income  = defaultdict(lambda: 0)
    wtp_health  = defaultdict(lambda: 0)

    ve = 0.7
    for phi in (.25, .50):
        for district in ["Chennai"]:
            if district == "Perambalur":
                    continue
            print(district)
            N_district = district_populations[district]
            dI_pc_range_nv = pd.read_csv(f"data/clean_sims/dT_TN_{district}_novaccination.csv")\
                .rename(columns = {"Unnamed: 0": "t"})\
                .set_index("t")/N_district
            dD_pc_range_nv = pd.read_csv(f"data/clean_sims/dD_TN_{district}_novaccination.csv")\
                .rename(columns = {"Unnamed: 0": "t"})\
                .set_index("t")/N_district
            Dx_nv = pd.read_csv(f"data/clean_sims/Dx_TN_{district}_novaccination.csv")\
                .drop(columns = ["Unnamed: 0"])\
                .reindex(range(1 + 5 * 365)).fillna(method = "ffill")            
            for policy in ["randomassignment", "mortalityprioritized", "contactrateprioritized"]:

                Dx_v  = pd.read_csv(f"data/clean_sims/Dx_TN_{district}_{policy}_ve70_annualgoal{int(100*phi)}_Rt_threshold0.2.csv")\
                    .drop(columns = ["Unnamed: 0"])\
                    .reindex(range(1 + 5 * 365)).fillna(method = "ffill")
    
                wtp_range[(phi, policy)] += np.array([
                    discounted_WTP(daily_WTP_old(district, N_district, dI_pc_range_nv.iloc[:, _], dD_pc_range_nv.iloc[:, _], Dx_v, Dx_nv)) 
                    for _ in range(100)
                ])
            # for district in ["Chennai"]:#, "Thiruvallur", "Vellore", "Viluppuram", "Tiruchirappalli", "Pudukkottai", "Thanjavur", "Dindigul", "Thiruvarur", "Theni"]:
                
                # dI_pc_range_v = pd.read_csv(f"data/clean_sims/dT_TN_{district}_{policy}_ve70_annualgoal{int(100*phi)}_Rt_threshold0.2.csv")\
                #     .rename(columns = {"Unnamed: 0": "t"})\
                #     .set_index("t")/N_district
                # dD_pc_range_v = pd.read_csv(f"data/clean_sims/dD_TN_{district}_{policy}_ve70_annualgoal{int(100*phi)}_Rt_threshold0.2.csv")\
                #     .rename(columns = {"Unnamed: 0": "t"})\
                #     .set_index("t")/N_district

        # wtp_daily[district] = daily_WTP(district, N_district, dI_pc_range.mean(axis = 1), dD_pc_range.mean(axis = 1), Dx_v, Dx_nv)
        # wtp_total[district] = discounted_WTP(wtp_daily[district])
        # wtp_range[district] = np.array([
        #         discounted_WTP(daily_WTP(district, N_district, dI_pc_range.iloc[:, _], dD_pc_range.iloc[:, _], Dx_v, Dx_nv)) 
        #         for _ in range(100)
        #     ])
        # wtp_percentiles[district] = np.percentile(wtp_range[district], [5, 50, 95], axis = 0)

        # dI_pc_range_v = pd.read_csv(f"data/latest_simes/dT_TN_{district}_randomassignment_ve70_annualgoal50_Rt_threshold0.2.csv")\
        #     .rename(columns = {"Unnamed: 0": "t"})\
        #     .set_index("t")/N_district
        # dD_pc_range_v = pd.read_csv(f"data/latest_simes/dD_TN_{district}_randomassignment_ve70_annualgoal50_Rt_threshold0.2.csv")\
        #     .rename(columns = {"Unnamed: 0": "t"})\
        #     .set_index("t")/N_district
        # wtp_health[district] = discounted_WTP(daily_wtp_health(district, N_district, dI_pc_range["0"], dD_pc_range["0"], dI_pc_range_v["0"], dD_pc_range_v["0"], Dx_v))

    wtp_percentiles = {k: np.percentile(np.sum(v, axis = 1), [5, 50, 95]) for (k, v) in wtp_range.items()}


    fig = plt.figure()
    for (i, phi) in ((0, 0.25), (1, 0.5)):
        for (dx, policy) in enumerate(["randomassignment", "mortalityprioritized", "contactrateprioritized"]):
            clr = [contactrate_vax_color,  mortality_vax_color, random_vax_color,][dx]
            (lo, md, hi) = wtp_percentiles[(phi, policy)]
            lo, md, hi = lo * USD, md * USD, hi * USD
            *_, bars = plt.errorbar(
                x = [i - 0.3* (1 - dx)], y = [md], yerr = [[md - lo], [hi - md]],
                figure = fig, fmt = "o", color = clr, label = None if i > 0 else ["random assignment", "mortality prioritized", "contact rate prioritized"][dx],
                ms = 12, elinewidth = 5
            )
            [_.set_alpha(0.5) for _ in bars]
    plt.legend(ncol = 3, fontsize = "20")
    plt.xticks([0, 1], ["$\phi = 25$%", "$\phi = 50$%"], fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.PlotDevice().ylabel("WTP (USD)\n")
    # plt.ylim(top = 900)
    plt.show()

    # statewide aggregation
    summed_WTP = [v.sum(axis = 0) for v in wtp_range.values()] 
    state_percentiles = np.percentile(summed_WTP, [5, 50, 95], axis = 0)
    fig = plt.figure()
    for age in range(7):
        plt.errorbar(
            x = [age],
            y = state_percentiles[1, age] * USD,
            yerr = [
                [(state_percentiles[1, age] - state_percentiles[0, age])*USD], 
                [(state_percentiles[2, age] - state_percentiles[1, age])*USD], 
            ], 
            color = age_group_colors[age],
            label = age_bin_labels[age],
            fmt = "o",
            figure = fig
        )
    plt.legend(title = "age group", loc = "lower right")
    plt.xticks([])
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.show()


    summed_wtp_health = sum(wtp_health.values()).round(2)
    summed_wtp_income = sum(wtp_income.values()).round(2)

    fig, ax = plt.subplots()
    ax.bar(summed_wtp_income.keys(), summed_wtp_income.values * USD, bottom = summed_wtp_health.values * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2)
    ax.bar(summed_wtp_health.keys(), summed_wtp_health.values * USD,                                          color = age_group_colors, edgecolor = age_group_colors, linewidth = 2)
    ax.bar([summed_wtp_income.keys()[0]], [0], label = "income", color = "white", edgecolor = "black", linewidth = 2)
    ax.bar([summed_wtp_income.keys()[0]], [0], label = "health", color = "black", edgecolor = "black", linewidth = 2)
    label = "health",
    plt.legend()
    # plt.ylim(4000, 14000)
    plt.PlotDevice().ylabel("WTP (USD)\n")
    plt.semilogy()
    plt.show()


    ordering_values = {k: v[1].max() for (k, v) in wtp_range.items()}
    district_ordering = sorted(ordering_values, key = ordering_values.get, reverse = True)


    # agg by district 
    fig = plt.figure()
    for (i, district) in enumerate(district_ordering):
        wtps = wtp_percentiles[i]
        for j in range(7):
            plt.errorbar(
                x = [i + 0.1 * (j - 3)],
                y = wtps[1, 6-j] * USD,
                yerr = [
                    [(wtps[1, 6-j] - wtps[0, 6-j]) * USD],
                    [(wtps[2, 6-j] - wtps[1, 6-j]) * USD]
                ], 
                fmt = "o",
                color = age_group_colors[6-j],
                figure = fig,
                label = None if i > 0 else age_bin_labels[6-j]
            )
    plt.xticks(
        range(len(district_ordering)),
        district_ordering,
        rotation = 90
    )
    plt.legend(title = "age group")
    # plt.ylim(0, 10000)
    plt.xlim(-0.5, len(district_ordering) - 0.5)
    plt.PlotDevice().ylabel("WTP (USD)")
    plt.show()


    ["30323d","3f414f","4d5061","55688f","5c80bc","7995be","95a9c0","b38d97","d5aca9"]
    # agg by age 
    fig = plt.figure()
    for age_index in range(7):
        for (district, color) in zip(district_ordering[::3], ["30323d","b27c66","4d5061","5aaa95","5c80bc","86a873","95a9c0","5cab7d","b38d97"]):
            plt.scatter(
                y = age_index,
                x = wtp_percentiles[district][1, age_index] * USD, 
                color = "#" + color,
                figure = fig,
                marker = "D",
                s = 100, 
                label = None if age_index > 0 else district
            )
    plt.yticks(range(7), age_bin_labels)
    plt.PlotDevice().xlabel("\nWTP (USD)").ylabel("age bin\n")
    plt.legend()
    plt.show()
