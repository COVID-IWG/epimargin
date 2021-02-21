import pandas as pd
from studies.age_structure.commons import *

""" Calculate willingness to pay """ 

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

# bin cutoffs for prevalence categories
datareg = pd.read_stata("data/datareg.dta")
dI_percap_cat_limits = [datareg[datareg.I_cat == _].I.max() for _ in range(11)]
dD_percap_cat_limits = [datareg[datareg.D_cat == _].D.max() for _ in range(11)]

dI_percap_cat_national_limits = [datareg[datareg.I_cat_national == _].I_national.max() for _ in range(11)]
dD_percap_cat_national_limits = [datareg[datareg.D_cat_national == _].D_national.max() for _ in range(11)]

# national prevalence categories are sparse and easily mapped as a function of time
def date_to_I_cat_natl(date: pd.Timestamp):
    if date.year != 2021:
        return 0
    return {
        1: 10, 
        2: 4, # should be 6 but it's missing in the cuts
        3: 3,
        4: 2,
        5: 2,
        6: 1,
        7: 1
    }.get(date.month, 0)

def coeff_label(suffix, dropped):
    return lambda i: str(i) + ("b" if i == dropped else "") + "." + suffix

def coeff_value(column, dropped):
    return lambda _: coeffs.loc[_[column].apply(lambda i: str(i) + ("b" if i == dropped else "") + "." + column)].value.values

def estimate_consumption_decline(district, dI_pc, dD_pc):
    district_code = str(district_codes[district])
    district_coeff_label = district_code + ("b" if district_code == "92" else "") + ".districtnum"
    constant = coeffs.loc["_cons"].value + coeffs.loc[district_coeff_label].value

    # map values to categoricals 
    indicators = pd.DataFrame({
        "I_cat": pd.cut(dI_pc, bins = dI_percap_cat_limits, labels = False).fillna(0).astype(int),
        "D_cat": pd.cut(dD_pc, bins = dD_percap_cat_limits, labels = False).fillna(0).astype(int),
        "date": [(simulation_start + pd.Timedelta(_, "days")) for _ in range(len(dI_pc))]
    }).assign(
        month          = lambda _: _.date.dt.month,
        I_cat_national = lambda _: _.date.apply(date_to_I_cat_natl)
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

def daily_WTP(district):
    ve  = 0.7 
    N_district = district_populations[district]

    # no vaccination case
    dI_pc = pd.read_csv(f"data/full_sims/dT_TN_{district}_novaccination.csv").rename(columns = {"Unnamed: 0": "t"}).set_index("t").mean(axis = 1)/N_district
    dD_pc = pd.read_csv(f"data/full_sims/dD_TN_{district}_novaccination.csv").rename(columns = {"Unnamed: 0": "t"}).set_index("t").mean(axis = 1)/N_district

    f_hat_nv = estimate_consumption_decline(district, dI_pc, dD_pc)
    consumption_nv = (1 + f_hat_nv)[:, None] * consumption_2019.loc[district].values

    Dx_nv = pd.read_csv(f"data/compartment_counts/Dx_TN_{district}_novaccination.csv").drop(columns = ["Unnamed: 0"])
    P_death_nv = Dx_nv.diff().fillna(0).cumsum()/split_by_age(N_district) 
    WTP_nv = ((1 - P_death_nv) * consumption_nv).set_index(f_hat_nv.index)

    # vaccination 
    f_hat_v = estimate_consumption_decline(district, pd.Series([0]), pd.Series([0]))
    consumption_v = (1 + f_hat_v)[:, None] * consumption_2019.loc[district].values

    Dx_v = pd.read_csv(f"data/compartment_counts/Dx_TN_{district}_randomassignment_ve70_annualgoal50_Rt_threshold0.2.csv").drop(columns = ["Unnamed: 0"])
    P_death_v = Dx_v.diff().fillna(0).cumsum()/split_by_age(N_district) 
    WTP_v = ((ve + (1-ve)*(1 - P_death_v)) * consumption_v).set_index(
        pd.date_range(start = simulation_start, end = simulation_start + pd.Timedelta(len(Dx_v) - 1, "days"))
    )

    return (WTP_v - WTP_nv)\
        .fillna(0) # where vaccination time series is longer than no vax timeseries, we assume consumption is identical 

# satej's method
def discounted_WTP(wtp, rate = (4.25/100)/365):
    (wtp * (1/np.power((1 + rate), np.arange(len(wtp))))[:, None]).sum(axis = 0)

# alice's method
def avg_monthly_WTP(wtp):
    return 30 * wtp.groupby(wtp.index.month.astype(str).str.zfill(2) + "_" + wtp.index.year.astype(str)).mean().mean()

def latex_table_row(rowname, items):
    return " & ".join([rowname] + items.round(2)) + " \\ "

# run calculations and then print out each aggregation
daily_WTP_calcs = {district: daily_WTP(district) for district in district_codes.keys()}

for (district, daily_wtp) in daily_WTP_calcs.items():
    try:
        print(latex_table_row(district, discounted_WTP(daily_WTP)))
    except:
        pass

for (district, daily_wtp) in daily_WTP_calcs.items():
    try:
        print(latex_table_row(district, avg_monthly_WTP(daily_WTP)))
    except:
        pass
