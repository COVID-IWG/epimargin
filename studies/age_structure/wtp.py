import pandas as pd
from studies.age_structure.commons import *

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
datareg = pd.read_stata("data/datareg.dta")

datareg.filter(regex = "I_.*national", axis = 1)\
    .sort_values("I_national")\
    .drop_duplicates(subset = ["I_cat_national"], keep = "last") 

datareg = datareg[datareg.month_code >= "January 01, 2020"]
dI_percap_cat_national_limits = [datareg[datareg.I_cat_national == _].I_national.max() for _ in range(11)]
dD_percap_cat_national_limits = [datareg[datareg.D_cat_national == _].D_national.max() for _ in range(11)]

datareg = datareg[datareg.month_code >= "March 01, 2020"]
dI_percap_cat_limits = [datareg[datareg.I_cat == _].I.max() for _ in range(11)]
dD_percap_cat_limits = [datareg[datareg.D_cat == _].D.max() for _ in range(11)]

infection_cutoffs = [
    1.31359918148e-06,
    2.78728440853e-06,
    5.98520919658e-06,
    9.01749487694e-06,
    .0000138806432497,
    .0000232067541053,
    .0000348692029503,
    .0000553322569194,
    .0000837807402432
]

death_cutoffs = [
    4.86518215364e-08,
    7.74357252506e-08,
    1.17110323077e-07,
    1.79850716711e-07,
    3.08246710742e-07,
    4.38650091191e-07,
    6.63577948309e-07,
    9.89375901681e-07,
    1.52713164555e-06
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

def estimate_consumption_decline_monthly(district, dI_pc, dD_pc):
    district_code = str(district_codes[district])
    district_coeff_label = district_code + ("b" if district_code == "92" else "") + ".districtnum"
    constant = coeffs.loc["_cons"].value + coeffs.loc[district_coeff_label].value

    timeseries = pd.DataFrame({ 
        "dI_pc": dI_pc.reindex(range(365), fill_value = 0), 
        "dD_pc": dD_pc.reindex(range(365), fill_value = 0), 
        "date": [(simulation_start + pd.Timedelta(_, "days")) for _ in range(365)]
    }).assign(
        month = lambda _: _.date.dt.strftime("%m_%Y")
    ).groupby("month")\
    .mean()

    return timeseries.reset_index()\
    .rename(columns = {"month": "month_code"})\
    .assign(
        I_cat = cut("dI_pc", infection_cutoffs),
        D_cat = cut("dD_pc",     death_cutoffs),
        month_timestamp = lambda _: pd.to_datetime(_.month_code, format = "%m_%Y"), 
        I_cat_national = lambda _: _.month_timestamp.apply(date_to_I_cat_natl),
        month = lambda _: _.month_code.str[:2].astype(int)
    ).assign(
        I_coeff     = coeff_value("I_cat",          dropped = 0),
        D_coeff     = coeff_value("D_cat",          dropped = 0),
        I_nat_coeff = coeff_value("I_cat_national", dropped = 0),
        month_coeff = coeff_value("month",          dropped = 1)
    ).rename(columns = {"month_code": "date"})\
    .set_index("date")\
    .filter(like = "coeff", axis = 1)\
    .sum(axis = 1) + constant

def daily_WTP(district):
    ve  = 0.7 
    N_district = district_populations[district]

    # no vaccination case
    dI_pc = pd.read_csv(f"data/full_sims/dT_TN_{district}_novaccination.csv")\
        .rename(columns = {"Unnamed: 0": "t"})\
        .set_index("t")\
        .mean(axis = 1)/N_district
    dD_pc = pd.read_csv(f"data/full_sims/dD_TN_{district}_novaccination.csv")\
        .rename(columns = {"Unnamed: 0": "t"})\
        .set_index("t")\
        .mean(axis = 1)/N_district

    f_hat_nv = estimate_consumption_decline(district, dI_pc, dD_pc)
    consumption_nv = (1 + f_hat_nv)[:, None] * consumption_2019.loc[district].values

    Dx_nv = pd.read_csv(f"data/compartment_counts/Dx_TN_{district}_novaccination.csv")\
        .drop(columns = ["Unnamed: 0"])
    P_death_nv = Dx_nv.diff().fillna(0).cumsum()/split_by_age(N_district) 
    WTP_nv = ((1 - P_death_nv) * consumption_nv).set_index(f_hat_nv.index)

    # vaccination 
    f_hat_v = estimate_consumption_decline(district, pd.Series([0]), pd.Series([0]))
    consumption_v = (1 + f_hat_v)[:, None] * consumption_2019.loc[district].values

    Dx_path = f"data/compartment_counts/Dx_TN_{district}_randomassignment_ve70_annualgoal50_Rt_threshold0.2.csv"
    Dx_v = pd.read_csv(Dx_path)\
        .drop(columns = ["Unnamed: 0"])
    P_death_v = Dx_v.diff().fillna(0).cumsum()/split_by_age(N_district) 
    WTP_v = ((ve + (1-ve)*(1 - P_death_v)) * consumption_v).set_index(
        pd.date_range(
            start = simulation_start, 
            end = simulation_start + pd.Timedelta(len(Dx_v) - 1, "days"))
    )

    # where vaccination time series is longer than no vax timeseries, we assume consumption is identical 
    return (WTP_v - WTP_nv).fillna(0) 

def monthly_WTP(district):
    
    Dx_nv.pipe(day_idx).groupby(lambda _:_.month).mean().pipe(month_idx)/N_Chennai
    Dx_v .pipe(day_idx).groupby(lambda _:_.month).mean().pipe(month_idx)/N_Chennai


def monthly_WTP_diffprob(district):
    Dx_nv\
        .diff().shift(-1).fillna(0).cumsum()\
        .pipe(day_idx).groupby(lambda _:_.month)\
        .mean().pipe(month_idx)/N_Chennai 
    Dx_v\
        .diff().shift(-1).fillna(0).cumsum()\
        .pipe(day_idx).groupby(lambda _:_.month)\
        .mean().pipe(month_idx)/N_Chennai 

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

# run calculations and then print out each aggregation
daily_WTP_calcs = {district: daily_WTP(district) for district in district_codes.keys()}

for (district, wtp) in daily_WTP_calcs.items():
    print(latex_table_row(district, discounted_WTP(wtp)/30))

for (district, wtp) in daily_WTP_calcs.items():
    print(latex_table_row(district, avg_monthly_WTP(wtp)))

