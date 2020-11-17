from logging import getLogger
from pathlib import Path

import adaptive.plots as plt
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.models import SIR, NetworkedSIR
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup, weeks, million
from tqdm import tqdm

logger = getLogger("SULSEL")

CI = 0.95
gamma = 0.2
window = 5
smoothing = notched_smoothing(window = window)

# load case data

schema = { 
    "Date Symptom Onset"          : "symptom_onset",
    "Date of Hospital  Admissions": "admission",
    "Date tested"                 : "tested",
    "Date of positive test result": "confirmed",
    "Date Recovered"              : "recovered",
    "Date Died"                   : "died",
    "Kebupaten/Kota"              : "regency",
    "Kecamatan"                   : "district",
    "age "                        : "age"
}

regency_names = { 
    'Pangkep'  : 'Pangkajene Dan Kepulauan', 
    'Pare-Pare': 'Parepare', 
    'Selayar'  : 'Kepulauan Selayar', 
    'Sidrap'   : 'Sidenreng Rappang'
}

def parse_datetimes(df):
    valid_idx = ~df.isna() & df.str.endswith("20")
    valid = df[valid_idx]
    monthfirst_idx = valid.str.endswith("/20") # short years -> month first notation 
    valid.loc[( monthfirst_idx)] = pd.to_datetime(valid[( monthfirst_idx)], errors = 'coerce', format = "%m/%d/%y", dayfirst = False)
    valid.loc[(~monthfirst_idx)] = pd.to_datetime(valid[(~monthfirst_idx)], errors = 'coerce', format = "%d/%m/%Y", dayfirst = True)
    # assert df.max() <= pd.to_datetime("October 03, 2020"), "date parsing resulted in future dates"
    df.loc[valid_idx] = valid.apply(pd.Timestamp)

def parse_age(s, bound = 100):
    try: 
        return min(bound, int(float(s.strip().split(" ")[0])))
    except:
        return None
    
cases = pd.read_csv("data/1 Nop 2020 Data collection template update South Sulawesi_update (01112020) (2).csv", usecols = schema.keys())\
        .rename(columns = schema)\
        .dropna(how = 'all')
parse_datetimes(cases.loc[:, "confirmed"])
cases.regency = cases.regency.str.title().map(lambda s: regency_names.get(s, s))
cases.age     = cases.age.apply(parse_age)
cases = cases.query("regency == 'Makassar'").dropna(subset = ["age"])
cases["age_bin"] = pd.cut(cases.age, [0, 20, 100], labels = ["school", "nonschool"])
cases = cases[cases.confirmed <= "Oct 25, 2020"]

age_ts = cases[["age_bin", "confirmed"]].groupby(["age_bin", "confirmed"]).size().sort_index()

(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(age_ts.loc["school"], CI = CI, smoothing = smoothing, totals = False)

school_Rt = np.mean(Rt_pred[-14:])
school_T_lb = T_CI_lower[-1]
school_T_ub = T_CI_upper[-1]

plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI)\
    .title("\nMakassar: Reproductive Number Estimate: school-age population")\
    .xlabel("\ndate")\
    .ylabel("$R_t$\n", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(age_ts.loc["nonschool"], CI = CI, smoothing = smoothing, totals = False)

plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI)\
    .title("\nMakassar: Reproductive Number Estimate: non-school-age population")\
    .xlabel("\ndate")\
    .ylabel("$R_t$\n", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

nonschool_Rt = np.mean(Rt_pred[-14:])
nonschool_T_lb = T_CI_lower[-1]
nonschool_T_ub = T_CI_upper[-1]

# set up models
total_pop = 1.339 * million
school_proportion = 0.25

# get empirical school beta 
dT = age_ts["school"][-2] 
b  = dT/age_ts["school"][-3]
I  = age_ts.sum(level = 0)["school"]
N  = school_proportion * total_pop
S  = N - I

beta = dT * N/(b * S * I)

def models(seed, simulated_school_Rt = school_Rt):
    school    = SIR("school",
        population = school_proportion * total_pop, 
        infectious_period = 6.25, 
        I0 = age_ts.sum(level = 0)["school"], 
        dT0 = age_ts.loc["school"][-1], 
        lower_CI = school_T_lb, 
        upper_CI = school_T_ub,
        Rt0 = simulated_school_Rt,
        random_seed = seed)
    nonschool = SIR("nonschool", 
        population = (1 - school_proportion) * total_pop, 
        infectious_period = 5.0, 
        I0 = age_ts.sum(level = 0)["nonschool"], 
        dT0 = age_ts.loc["nonschool"][-1], 
        lower_CI = school_T_lb, 
        upper_CI = school_T_ub,
        Rt0 = nonschool_Rt,
        random_seed = seed)
    return (school, nonschool)

def run_simulation(school: SIR, nonschool: SIR):
    for _ in range(30):
        school.run(1)
        nonschool.forward_epi_step(dB = 2*school.dT[-1]//3) # doesn't preserve population

    dT_school    = np.array(school.dT)
    dT_nonschool = np.array(nonschool.dT)

    dT = dT_school + dT_nonschool

    var_up_school = np.array(school.upper_CI) - dT_school
    var_dn_school = np.array(school.lower_CI) - dT_school

    var_up_nonschool = np.array(nonschool.upper_CI) - dT_nonschool
    var_dn_nonschool = np.array(nonschool.lower_CI) - dT_nonschool

    dT_CI_l = dT - np.sqrt(var_dn_school**2 + var_dn_nonschool**2)
    dT_CI_u = dT + np.sqrt(var_up_school**2 + var_up_nonschool**2)

    return (dT[1:], dT_CI_l[1:], dT_CI_u[1:])


rt1_10x = run_simulation(*models(0, 1.10*school_Rt))
rt1_25x = run_simulation(*models(0, 1.25*school_Rt))
rt1_50x = run_simulation(*models(0, 1.50*school_Rt))

(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(age_ts.sum(level = 1), CI = CI, smoothing = smoothing, totals = False)

MAK = SIR(name = "MAK", population = 1.339e6, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], mobility = 0, random_seed = 0, I0 = age_ts.sum(level = 1).sum()).run(30)

plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
    prediction_ts = [
        (MAK.dT[:-1], MAK.lower_CI[1:], MAK.upper_CI[1:], plt.PRED_PURPLE, "current social distancing"),
        (*rt1_10x, "orange",         "10% increase in school-age $R_t$"),
        (*rt1_25x, "mediumseagreen", "25% increase in school-age $R_t$"),
        (*rt1_50x, "hotpink",        "50% increase in school-age $R_t$"),
    ])\
    .xlabel("\ndate")\
    .ylabel("cases\n")
    # .title("\nMakassar Daily Cases - school reopening scenarios")\
    # .annotate("\nBayesian training process on empirical data, with anomalies identified")
(_, r) = plt.xlim()
plt.xlim(left = pd.Timestamp("Sep 1, 2020"), right = r)
plt.ylim(bottom = 10, top = 1000)
plt.vlines(dates[-1], ymin = 1, ymax = 1000, color = "black", linestyles = "solid")
plt.semilogy()
plt.show()