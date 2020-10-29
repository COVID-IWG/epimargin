import adaptive.plots as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data
from adaptive.models import SIR, NetworkedSIR
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd, days, weeks
from tqdm import tqdm

# model details
CI        = 0.95
smoothing = 7
gamma     = 0.2

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 18)]
}

for target in paths['v3'] + paths['v4']:
    try: 
        download_data(data, target)
    except:
        pass 

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts_full = get_time_series(df, "detected_state")
ts = ts_full.query("status_change_date <= 'October 14, 2020'")

states    = ["Bihar", "Maharashtra", "Punjab", "Tamil Nadu"][-1:]
codes     = ["BR",    "MH",          "PN",     "TN"][-1:]
pops      = [99.02e6, 114.2e6,       27.98e6,  67.86e6][-1:]
Rt_ranges = [(0.9, 2.4), (0.55, 2.06), (0.55, 2.22), (0.84, 1.06)][-1:]
windows   = [7, 14, 7, 10][-1:]


for (state, code, pop, Rt_range, smoothing) in zip(states, codes, pops, Rt_ranges, windows): 
    print(state)
    print("  + running estimation...")
    state_ts_full = pd.Series(data = notched_smoothing(window = smoothing)(ts_full.loc[state].Hospitalized), index = ts_full.loc[state].Hospitalized.index)
    (dates, Rt_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
        = analytical_MPVS(ts.loc[state].Hospitalized, CI = CI, smoothing = lambda x:x, totals = False)
    Rt = pd.DataFrame({"Rt": Rt_pred}, index = dates)
    Rt_m = np.mean(Rt[(Rt.index >= "31 March, 2020") & (Rt.index <= "17 May, 2020")])[0]
    Rt_v = np.mean(Rt[(Rt.index <  "31 March, 2020")])[0]
    print("  + Rt today:", Rt_pred[-1])
    print("  + Rt_m    :", Rt_m)
    print("  + Rt_v    :", Rt_v)
    historical = pd.DataFrame({"smoothed": new_cases_ts}, index = dates)

    plt.Rt(dates, Rt_pred, RR_CI_lower, RR_CI_upper, CI)\
        .ylabel("$R_t$")\
        .xlabel("date")\
        .title(f"\n{state}: Reproductive Number Estimate")\
        .annotate(f"public data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}")\
        .show()
    
    I0 = (ts.loc[state].Hospitalized - ts.loc[state].Recovered - ts.loc[state].Deceased).sum()
    state_model = SIR(name = state, population = pop, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], mobility = 0, I0 = I0, upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], random_seed = 0).run(10)

    empirical = state_ts_full[(state_ts_full.index >= "Oct 14, 2020") & (state_ts_full.index < "Oct 25, 2020")]

    plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
        prediction_ts=[
            (state_model.dT, state_model.lower_CI, state_model.upper_CI, plt.PRED_PURPLE, "predicted cases"),
        ] + [(empirical, empirical, empirical, "black", "empirical post-prediction cases")] if state != "Maharashtra" else [])\
        .ylabel("cases")\
        .xlabel("date")\
        .title(f"\n{state}: Daily Cases")\
        .annotate("\nBayesian training process on empirical data, with anomalies identified")
    _, r = plt.xlim()
    plt.xlim(left = pd.Timestamp("August 01, 2020"), right = r)
    plt.show()
    
    gdf = gpd.read_file(data/f"{code}.json")
    district_names = sorted(gdf.district)
    district_time_series = get_time_series(df[df.detected_state == state], "detected_district").Hospitalized
    migration = np.zeros((len(district_names), len(district_names)))
    estimates = []
    max_len = 1 + max(map(len, district_names))
    with tqdm(district_time_series.index.get_level_values(0).unique()) as districts:
        for district in districts:
            districts.set_description(f"{district :<{max_len}}")
            try: 
                (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, *_) = analytical_MPVS(district_time_series.loc[district], CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
                estimates.append((district, Rt_pred[-1], Rt_CI_lower[-1], Rt_CI_upper[-1], linear_projection(dates, Rt_pred, smoothing, period = 7))) 
            except (IndexError, ValueError): 
                estimates.append((district, np.nan, np.nan, np.nan, np.nan))
    estimates = pd.DataFrame(estimates).dropna()
    estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
    estimates.set_index("district", inplace=True)
    estimates = estimates.clip(0)
    estimates.to_csv(data/f"Rt_estimates_{code}.csv")
    print(estimates)
    print(estimates.Rt.max(), estimates.Rt_proj.max())
    print(estimates.Rt.min(), estimates.Rt_proj.min())

    gdf = gdf.merge(estimates, left_on = "district", right_on = "district")
    plt.choropleth(gdf, mappable = plt.get_cmap(*Rt_range))\
       .title(f"\n{state}: $R_t$ by District")\
       .show()

    if state == "Punjab":
        def model(seed):
            return NetworkedSIR([SIR(name = state, population = pop, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], mobility = 0, I0 = I0)], random_seed = seed) 
        
        def run_policies(seed, lockdown_period = 7 * days, total = 45 * days):
            model_A = model(seed)
            simulate_lockdown(model_A, lockdown_period, total, {state: Rt_m}, {state: Rt_v}, np.zeros((1, 1)), np.zeros((1, 1)))

            # lockdown 1
            model_B = model(seed)
            simulate_lockdown(model_B, lockdown_period + 2*weeks, total, {state: Rt_m}, {state: Rt_v}, np.zeros((1, 1)), np.zeros((1, 1)))

            # lockdown + adaptive controls
            model_C = model(seed)
            simulate_adaptive_control(model_C, lockdown_period + 2*weeks, total, np.zeros((1, 1)), np.zeros((1, 1)), 
            {state: Rt_m}, {state: gamma * Rt_v}, {state: gamma * Rt_m})

            return model_A, model_B, model_C
        
        si, sf = 0, 500
        simulation_results = [run_policies(seed) for seed in tqdm(range(si, sf))]
        plt.simulations(simulation_results, 
            ["28 October: Lockdown Release", "11 November: Lockdown Release", "11 November: Start Adaptive Control"], 
            historical = historical[historical.index >= "01 June, 2020"])\
            .title(f"\n{state} Policy Scenarios: Projected Cases over Time")\
            .xlabel("date")\
            .ylabel("cases")\
            .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {smoothing}")\
            .show()

    
# national 
state = "India"
ts = get_time_series(df)

smoothing = 14
ts_full = pd.Series(data = notched_smoothing(window = smoothing)(ts.Hospitalized), index = ts.index)
ts_trunc = ts_full[ts_full.index <= 'October 14, 2020']
(dates, Rt_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(ts_trunc, CI = CI, smoothing = lambda x:x, totals = False)
Rt = pd.DataFrame({"Rt": Rt_pred}, index = dates)
Rt_m = np.mean(Rt[(Rt.index >= "31 March, 2020") & (Rt.index <= "17 May, 2020")])[0]
Rt_v = np.mean(Rt[(Rt.index <  "31 March, 2020")])[0]
print("  + Rt today:", Rt_pred[-1])
print("  + Rt_m    :", Rt_m)
print("  + Rt_v    :", Rt_v)
historical = pd.DataFrame({"smoothed": new_cases_ts}, index = dates)

# plt.Rt(dates, Rt_pred, RR_CI_lower, RR_CI_upper, CI)\
#     .ylabel("$R_t$")\
#     .xlabel("date")\
#     .title(f"\n{state}: Reproductive Number Estimate")\
#     .annotate(f"public data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}")\
#     .show()

I0 = (ts[ts.index <= 'October 14, 2020'].Hospitalized - ts[ts.index <= 'October 14, 2020'].Recovered - ts[ts.index <= 'October 14, 2020'].Deceased).sum()
state_model = SIR(name = state, population = 1.3e9, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], mobility = 0, I0 = I0, upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], random_seed = 0).run(10)

empirical = ts_full[(ts_full.index >= "Oct 14, 2020") & (ts_full.index < "Oct 25, 2020")]

plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
    prediction_ts=[
        (state_model.dT, state_model.lower_CI, state_model.upper_CI, plt.PRED_PURPLE, "predicted cases"),
    ] + [(empirical, empirical, empirical, "black", "empirical post-prediction cases")] if state != "Maharashtra" else [])\
    .ylabel("cases")\
    .xlabel("date")\
    .title(f"\n{state}: Daily Cases")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")
_, r = plt.xlim()
plt.xlim(left = pd.Timestamp("August 01, 2020"), right = r)
plt.show()