from typing import Callable, Tuple
from adaptive.models import SIR
import pandas as pd

from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data
import adaptive.plots as plt
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd, weeks
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
from tqdm import tqdm

# model details
CI        = 0.95
smoothing = 7

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 26)]
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

# cutoff = None
# cutoff = "April 7, 2021"
cutoff = "April 14, 2021"

if cutoff:
    df = df[df.date_announced <= cutoff]
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(
    df[df.detected_state == "Tamil Nadu"], 
    ["detected_state", "detected_district"]
)\
.drop(columns = ["date", "time", "delta", "logdelta"])\
.rename(columns = {
            "Deceased":     "dD",
            "Hospitalized": "dT",
            "Recovered":    "dR"
}).droplevel(0)\
.drop(labels = ["Other State", "Railway Quarantine", "Airport Quarantine"])


district_estimates = []

simulation_initial_conditions = pd.read_csv(data/f"all_india_coalesced_initial_conditions{simulation_start.strftime('%b%d')}.csv")\
    .drop(columns = ["Unnamed: 0"])\
    .set_index(["state", "district"])\
    .loc["Tamil Nadu"]

def setup(district) -> Tuple[Callable[[str], SIR], pd.DataFrame]:
    demographics = simulation_initial_conditions.loc[district]
    
    dR_conf = ts.loc[district].dR
    dR_conf = dR_conf.reindex(pd.date_range(dR_conf.index.min(), dR_conf.index.max()), fill_value = 0)
    dR_conf_smooth = pd.Series(smooth(dR_conf), index = dR_conf.index).clip(0).astype(int)
    R_conf_smooth  = dR_conf_smooth.cumsum().astype(int)

    R0 = R_conf_smooth[data_recency]

    dD_conf = ts.loc[district].dD
    dD_conf = dD_conf.reindex(pd.date_range(dD_conf.index.min(), dD_conf.index.max()), fill_value = 0)
    dD_conf_smooth = pd.Series(smooth(dD_conf), index = dD_conf.index).clip(0).astype(int)
    D_conf_smooth  = dD_conf_smooth.cumsum().astype(int)
    D0 = D_conf_smooth[data_recency]

    dT_conf = ts.loc[district].dT
    dT_conf = dT_conf.reindex(pd.date_range(dT_conf.index.min(), dT_conf.index.max()), fill_value = 0)

    (
        dates,
        Rt_pred, Rt_CI_upper, Rt_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        *_
    ) = analytical_MPVS(ts.loc[district].dT, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
    Rt_estimates = pd.DataFrame(data = {
        "dates"       : dates,
        "Rt_pred"     : Rt_pred,
        "Rt_CI_upper" : Rt_CI_upper,
        "Rt_CI_lower" : Rt_CI_lower,
        "T_pred"      : T_pred,
        "T_CI_upper"  : T_CI_upper,
        "T_CI_lower"  : T_CI_lower,
        "total_cases" : total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })

    dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index).clip(0).astype(int)
    T_conf_smooth  = dT_conf_smooth.cumsum().astype(int)
    T0 = T_conf_smooth[data_recency]
    dT0 = dT_conf_smooth[data_recency]

    S0 = max(0, demographics.N_tot - T0)
    I0 = max(0, T0 - R0 - D0)

    return ( 
        lambda seed = 0: SIR(
            name = district, 
            mortality = demographics[[f"N_{i}" for i in range(7)]] @ np.array(list(TN_IFRs.values()))/demographics.N_tot,
            population = demographics.N_tot, 
            random_seed = seed,
            infectious_period = 10, 
            S0  = S0,
            I0  = I0, 
            R0  = R0, 
            D0  = D0, 
            dT0 = dT0, 
            Rt0 = Rt_estimates.set_index("dates").loc[data_recency].Rt_pred * demographics.N_tot/S0), 
        Rt_estimates
    )
        
district_estimates = []    
for district in tqdm(simulation_initial_conditions.index.get_level_values(0).unique()):
    simulation, Rt_estimates = setup(district)
    district_estimates.append(Rt_estimates.assign(district = district))
    Rt_estimates.to_csv(data/f"TN_Rt_data_{district}_{data_recency}_run{run_date}.csv")
    projections = pd.DataFrame(
        np.array(
            [simulation(_).run(6 * weeks).dT for _ in range(1000)]
    )).astype(int).T\
    .set_index(pd.date_range(start = data_recency, freq = "D", periods = 6*weeks + 1))
    print(district, projections.mean(axis = 1))
    projections.to_csv(data/f"TN_projections/projections_{district}_data{data_recency}_run{run_date}.csv")