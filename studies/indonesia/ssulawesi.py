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
from adaptive.utils import days, setup, weeks
from tqdm import tqdm

logger = getLogger("SULSEL")

CI = 0.95
gamma = 0.2
window = 5
smoothing = notched_smoothing(window = window)

schema = { 
    "Date Symptom Onset"          : "symptom_onset",
    "Date of Hospital  Admissions": "admission",
    "Date tested"                 : "tested",
    "Date of positive test result": "confirmed",
    "Date Recovered"              : "recovered",
    "Date Died"                   : "died",
    "Kebupaten/Kota"              : "regency",
    "Kecamatan"                   : "district"
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

    
cases = pd.read_csv("data/3 OCT 2020 Data collection template update South Sulawesi_CASE.csv", usecols = schema.keys())\
        .rename(columns = schema)\
        .dropna(how = 'all')
parse_datetimes(cases.loc[:, "confirmed"])
cases.regency = cases.regency.str.title().map(lambda s: regency_names.get(s, s))

# generation_interval = cases[~cases.symptom_onset.isna() & ~cases.confirmed.isna()]\
#     .apply(get_generation_interval, axis = 1)\
#     .dropna()\
#     .value_counts()\
#     .sort_index()
# generation_interval =  generation_interval[(generation_interval.index >= 0) & (generation_interval.index <= 60)]
# generation_interval /= generation_interval.sum()

new_cases = cases.confirmed.value_counts().sort_index()
new_cases_smoothed = smoothing(new_cases)
plt.plot(new_cases, '.', color = "blue")
plt.plot(new_cases.index, new_cases_smoothed, '-', color = "black")
plt.show()

logger.info("running province-level Rt estimate")
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(new_cases, CI = CI, smoothing = smoothing, totals = False)

plt.Rt(dates, Rt_pred[1:], Rt_CI_upper[1:], Rt_CI_lower[1:], CI)\
    .title("\nSouth Sulawesi: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$\n", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

logger.info("running case-forward prediction")
prediction_period = 14*days
I0 = (~cases.confirmed.isna()).sum() - (~cases.recovered.isna()).sum() - (~cases.died.isna()).sum()
IDN = SIR(name = "IDN", population = 8_819_500, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], mobility = 0, random_seed = 0, I0 = I0)\
           .run(prediction_period)

plt.daily_cases(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI, 
    prediction_ts = [
        (IDN.dT[:-1], IDN.lower_CI[1:], IDN.upper_CI[1:], plt.PRED_PURPLE, "predicted cases")
    ])\
    .title("\nSouth Sulawesi: Daily Cases")\
    .xlabel("\ndate")\
    .ylabel("cases\n")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")\
    .show()

# makassar estimates 
mak_cases     = cases[cases.regency == "Makassar"]
mak_new_cases = mak_cases.confirmed.value_counts().sort_index()
logger.info("running city-level Rt estimate")
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(mak_new_cases, CI = CI, smoothing = smoothing, totals = False)

plt.Rt(dates, Rt_pred[1:], Rt_CI_upper[1:], Rt_CI_lower[1:], CI)\
    .title("\nMakassar: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$\n", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

logger.info("running case-forward prediction")
prediction_period = 14*days
I0 = (~mak_cases.confirmed.isna()).sum() - (~mak_cases.recovered.isna()).sum() - (~mak_cases.died.isna()).sum()
MAK = SIR(name = "MAK", population = 1.339e6, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], mobility = 0, random_seed = 0, I0 = I0)\
           .run(prediction_period)

plt.daily_cases(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI, 
    prediction_ts = [
        (MAK.dT[:-1], MAK.lower_CI[1:], MAK.upper_CI[1:], plt.PRED_PURPLE, "predicted cases")
    ])\
    .title("\nMakassar: Daily Cases")\
    .xlabel("\ndate")\
    .ylabel("cases\n")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")\
    .show()

logger.info("regency-level projections")
regencies = sorted(cases.regency.unique())
regency_cases = cases.groupby(["regency", "confirmed"]).size().sort_index().unstack(fill_value = 0).stack() 
migration = np.zeros((len(regencies), len(regencies)))
estimates = []
max_len = 1 + max(map(len, regencies))
with tqdm(regencies) as progress:
    for regency in regencies:
        progress.set_description(f"{regency :<{max_len}}")
        (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, *_) = analytical_MPVS(regency_cases.loc[regency], CI = CI, smoothing = smoothing, totals=False)
        estimates.append((regency, Rt_pred[-1], Rt_CI_lower[-1], Rt_CI_upper[-1], linear_projection(dates, Rt_pred, 7)))
estimates = pd.DataFrame(estimates)
estimates.columns = ["regency", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("regency", inplace=True)
estimates.to_csv("data/SULSEL_Rt_projections.csv")
print(estimates)

gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_2.shp")\
    .query("NAME_1 == 'Sulawesi Selatan'")\
    .merge(estimates, left_on = "NAME_2", right_on = "regency")

choro = plt.choropleth(gdf, mappable = plt.get_cmap(0.4, 1.4))

for ax in choro.figure.axes[:-1]:
    plt.sca(ax)
    plt.xlim(left = 119, right = 122)
    plt.ylim(bottom = -7.56, top = -1.86)

plt.show()

logger.info("adaptive control")
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(new_cases, CI = CI, smoothing = smoothing, totals = False)
Rt = pd.DataFrame(data = {"Rt": Rt_pred[1:]}, index = dates)
Rt_current = Rt_pred[-1]
Rt_m = np.mean(Rt[(Rt.index >= "April 21, 2020") & (Rt.index <= "May 22, 2020")])[0]
Rt_v = np.mean(Rt[(Rt.index <= "April 14, 2020")])[0]

Rt_m_scaled = Rt_current + 0.75 * (Rt_m - Rt_current)
Rt_v_scaled = Rt_current + 0.75 * (Rt_v - Rt_current)


historical = pd.DataFrame({"smoothed": smoothing(new_cases)}, index = new_cases.index)
def model(seed = 0):
    return NetworkedSIR([SIR(
        name = "SULSEL", population = 8_819_500, 
        dT0 = historical.iloc[-1][0], Rt0 = Rt_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], 
        mobility = 0, I0 = historical.sum()[0]
    )], random_seed = seed)

lockdown_period = 7 * days
total = 45 * days

def run_policies(seed):
    model_A = model(seed)
    simulate_lockdown(model_A, lockdown_period, total, {"SULSEL": Rt_m_scaled}, {"SULSEL": Rt_v_scaled}, np.zeros((1, 1)), np.zeros((1, 1)))

    # lockdown 1
    model_B = model(seed)
    simulate_lockdown(model_B, lockdown_period + 2*weeks, total, {"SULSEL": Rt_m_scaled}, {"SULSEL": Rt_v_scaled}, np.zeros((1, 1)), np.zeros((1, 1)))

    # lockdown + adaptive controls
    model_C = model(seed)
    simulate_adaptive_control(model_C, lockdown_period + 2*weeks, total, np.zeros((1, 1)), np.zeros((1, 1)), 
    {"SULSEL": Rt_m_scaled}, {"SULSEL": gamma * Rt_v_scaled}, {"SULSEL": gamma * Rt_m_scaled})

    return model_A, model_B, model_C

si, sf = 0, 1000
simulation_results = [run_policies(seed) for seed in tqdm(range(si, sf))]
plt.simulations(simulation_results, 
    ["07 October: Return to 75% Max Mobility", "21 October: Return to 75% Max Mobility", "21 October: Start Adaptive Control"], 
    historical = historical[historical.index >= "01 June, 2020"])\
    .title("\nSouth Sulawesi Policy Scenarios: Projected Cases over Time")\
    .xlabel("date")\
    .ylabel("cases")\
    .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}")\
    .show()