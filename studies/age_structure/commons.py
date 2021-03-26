from pathlib import Path

import numpy as np
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data, state_name_lookup
from adaptive.smoothing import notched_smoothing
from tqdm import tqdm

""" Common data loading/cleaning functions and constants """

data = Path("./data").resolve()

USD = 1/72

agebin_labels = ["0-17", "18-29","30-39", "40-49", "50-59", "60-69","70+"]

# Rt estimation parameters
CI = 0.95
window = 14
gamma = 0.2
infectious_period = 5
smooth = notched_smoothing(window)

# simulation parameters
simulation_start = pd.Timestamp("Jan 1, 2021")
num_sims = 100
focus_states = ["Tamil Nadu", "Punjab", "Maharashtra", "Bihar", "West Bengal"]

# common vaccination parameters
immunity_threshold = 0.75
Rt_threshold = 0.2

# misc
survey_date = "October 23, 2020"

# palette 
TN_color = "firebrick"
IN_color = "#292f36"

no_vax_color          = "black"
contactrate_vax_color = "darkorange"
random_vax_color      = "royalblue"
mortality_vax_color   = "forestgreen"

agebin_colors = [ "#05668d", "#427aa1", "#679436", "#a5be00", "#ffcb77", "#d0393b", "#7a306c"]

#################################################################

print(":: loading admin data")
# load admin data on population
IN_age_structure = { # WPP2019_POP_F01_1_POPULATION_BY_AGE_BOTH_SEXES
    "0-17":   116880 + 117982 + 126156 + 126046,
    "18-29":  122505 + 117397,
    "30-39":  112176 + 103460,
    "40-49":   90220 +  79440,
    "50-59":   68876 +  59256,
    "60-69":   48891 +  38260,
    "70+":     24091 +  15084 +   8489 +   3531 + 993 + 223 + 48,
}

TN_age_structure = { 
    "0-17" : 15581526,
    "18-29": 15674833,
    "30-39": 11652016,
    "40-49":  9777265,
    "50-59":  6804602,
    "60-69":  4650978,
    "70+":    2858780,
}

N_j = np.array([20504724, 15674833, 11875848, 9777265, 6804602, 4650978, 2858780])

TN_IFRs = { 
    "0-17" : 0.00003,
    "18-29": 0.00003,
    "30-39": 0.00010,
    "40-49": 0.00032,
    "50-59": 0.00111,
    "60-69": 0.00264,
    "70+"  : 0.00588,
}

median_ages = { 
    "0-17" : 8.5,
    "18-29": 23.5,
    "30-39": 34.5,
    "40-49": 44.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "70+"  : 85,
}

district_IFR = pd.read_csv(data/"district_estimates.csv").set_index("district")
district_IFR.drop(columns = [_ for _ in district_IFR.columns if "Unnamed" in _], inplace = True)


# normalize
IN_age_structure_norm = sum(IN_age_structure.values())
IN_age_ratios = np.array([v/IN_age_structure_norm for v in IN_age_structure.values()])

TN_age_structure_norm = sum(TN_age_structure.values())
TN_age_ratios = np.array([v/TN_age_structure_norm for v in TN_age_structure.values()])
split_by_age = lambda v: (v * TN_age_ratios).astype(int)

TN_IFR_norm = sum(TN_IFRs.values())
TN_IFR_ratios = np.array([v/TN_IFR_norm for v in TN_IFRs.values()])
split_by_IFR = lambda v: (v * TN_IFR_ratios).astype(int)

# redefined estimators
TN_death_structure = pd.Series({ 
    "0-17" : 32,
    "18-29": 121,
    "30-39": 368,
    "40-49": 984,
    "50-59": 2423,
    "60-69": 3471,
    "70+"  : 4339,
})


TN_recovery_structure = pd.Series({ 
    "0-17": 5054937,
    "18-29": 4819218,
    "30-39": 3587705,
    "40-49": 3084814,
    "50-59": 2178817,
    "60-69": 1313049,
    "70+": 738095,
})

TN_infection_structure = TN_death_structure + TN_recovery_structure
fS = pd.Series(TN_age_ratios)[:, None]
fD = (TN_death_structure     / TN_death_structure    .sum())[:, None]
fR = (TN_recovery_structure  / TN_recovery_structure .sum())[:, None]
fI = (TN_infection_structure / TN_infection_structure.sum())[:, None]

india_pop = pd.read_csv(data/"india_pop.csv", names = ["state", "population"], index_col = "state").to_dict()["population"]

def get_state_timeseries(states = "*", download: bool = False) -> pd.DataFrame:
    """ load state- and district-level data, downloading source files if specified """
    paths = {"v3": [data_path(i) for i in (1, 2)], "v4": [data_path(i) for i in range(3, 25)]}
    if download:
        for target in paths['v3'] + paths['v4']: 
            download_data(data, target)
    return load_all_data(v3_paths = [data/filepath for filepath in paths['v3']],  v4_paths = [data/filepath for filepath in paths['v4']])\
        .query("detected_state in @states" if states != "*" else "detected_state != 'NULL'", engine = "python")\
        .pipe(lambda _: get_time_series(_, ["detected_state", "detected_district"]))\
        .drop(columns = ["date", "time", "delta", "logdelta"])\
        .rename(columns = {
            "Deceased":     "dD",
            "Hospitalized": "dT",
            "Recovered":    "dR"
        })


def case_death_timeseries(states = "*", download = False):
    """ assemble a list of daily deaths and cases for consumption prediction """
    ts = get_state_timeseries(states, download)
    ts_index = pd.date_range(start = ts.index.get_level_values(-1).min(), end = ts.index.get_level_values(-1).max(), freq = "D")

    return ts.unstack(-1)\
        .fillna(0)\
        .stack()\
        .swaplevel(-1, 0)\
        .reindex(ts_index, level = 0, fill_value = 0)\
        .swaplevel(-1, 0)

def assemble_sero_data():
    district_sero = pd.read_stata(data/"seroprevalence_district.dta")\
        .rename(columns = lambda _:_.replace("_api", ""))\
        .sort_values(["state", "district", "agecat"])\
        .assign(agecat = lambda _: _["agecat"].astype(int))\
        .fillna(0)\
        .pivot_table(index = ["state", "district"], columns = "agecat", values = "seroprevalence")\
        .rename(columns = {i+1: f"sero_{i}" for i in range(7)})\
        .assign(sero_0 = lambda _:_["sero_1"])
    all_crosswalk = pd.read_stata(data/"all_crosswalk.dta")\
        .filter(regex = "tot_pop[0-9]$|.*_api", axis = 1)\
        .sort_values(["state_api", "district_api"])\
        .rename(columns = lambda _:_.replace("_api", ""))\
        .rename(columns = lambda _:_.replace("tot_pop", "N_"))\
        .rename(columns = {f"N_{i+1}": f"N_{i}" for i in range(7)})\
        .assign(N_tot = lambda _:_.filter(like = "N_", axis = 1).sum(axis = 1))\
        .set_index(["state", "district"])

    return district_sero.join(all_crosswalk)

def assemble_initial_conditions(states = "*", simulation_start = simulation_start, survey_date = survey_date):
    rows = []
    district_age_pop = pd.read_csv(data/"all_india_sero_pop.csv").set_index(["state", "district"])
    if states == "*":
        districts_to_run = district_age_pop
    else:
        districts_to_run = district_age_pop[district_age_pop.index.isin(states, level = 0)]

    progress = tqdm(total = 4 * len(districts_to_run) + 11)
    progress.set_description(f"{'loading case data':<20}")
    
    ts = get_state_timeseries(states)
    progress.update(10)
    for ((state, district), 
        sero_0, sero_1, sero_2, sero_3, sero_4, sero_5, sero_6, 
        N_0, N_1, N_2, N_3, N_4, N_5, N_6, N_tot
    ) in districts_to_run.itertuples():
        progress.set_description(f"{state[:20]:<20}")
        
        dR_conf = ts.loc[state, district].dR
        dR_conf = dR_conf.reindex(pd.date_range(dR_conf.index.min(), dR_conf.index.max()), fill_value = 0)
        if len(dR_conf) >= window + 1:
            dR_conf_smooth = pd.Series(smooth(dR_conf), index = dR_conf.index).clip(0).astype(int)
        else: 
            dR_conf_smooth = dR_conf

        R_conf_smooth  = dR_conf_smooth.cumsum().astype(int)
        R_conf = R_conf_smooth[survey_date if survey_date in R_conf_smooth.index else -1]
        R_sero = (sero_0*N_0 + sero_1*N_1 + sero_2*N_2 + sero_3*N_3 + sero_4*N_4 + sero_5*N_5 + sero_6*N_6)
        R_ratio = R_sero/R_conf 
        R0 = R_conf_smooth[simulation_start if simulation_start in R_conf_smooth.index else -1] * R_ratio
        progress.update(1)
        
        dD_conf = ts.loc[state, district].dD
        dD_conf = dD_conf.reindex(pd.date_range(dD_conf.index.min(), dD_conf.index.max()), fill_value = 0)
        if len(dD_conf) >= window + 1:
            dD_conf_smooth = pd.Series(smooth(dD_conf), index = dD_conf.index).clip(0).astype(int)
        else:
            dD_conf_smooth = dD_conf
        D_conf_smooth  = dD_conf_smooth.cumsum().astype(int)
        D0 = D_conf_smooth[simulation_start if simulation_start in D_conf_smooth.index else -1]
        progress.update(1)
        
        dT_conf = ts.loc[state, district].dT
        dT_conf = dT_conf.reindex(pd.date_range(dT_conf.index.min(), dT_conf.index.max()), fill_value = 0)
        if len(dT_conf) >= window + 1:
            dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index).clip(0).astype(int)
        else:
            dT_conf_smooth = dT_conf
        T_conf_smooth  = dT_conf_smooth.cumsum().astype(int)
        T_conf = T_conf_smooth[survey_date if survey_date in T_conf_smooth.index else -1]
        T_sero = R_sero + D0 
        T_ratio = T_sero/T_conf
        T0 = T_conf_smooth[simulation_start if simulation_start in T_conf_smooth.index else -1] * T_ratio
        progress.update(1)

        S0 = N_tot - T0
        dD0 = dD_conf_smooth[simulation_start if simulation_start in dD_conf_smooth.index else -1]
        dT0 = dT_conf_smooth[simulation_start if simulation_start in dT_conf_smooth.index else -1] * T_ratio
        I0 = max(0, (T0 - R0 - D0))

        (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_smooth, CI = CI, smoothing = lambda _:_, totals = False)
        Rt = dict(zip(Rt_dates, Rt_est))

        rows.append((state_name_lookup[state], state, district, 
            sero_0, N_0, sero_1, N_1, sero_2, N_2, sero_3, N_3, sero_4, N_4, sero_5, N_5, sero_6, N_6, N_tot, 
            Rt.get(simulation_start, Rt[max(Rt.keys())]), S0, I0, R0, D0, dT0, dD0
        ))
        progress.update(1)
    progress.set_description(f"{'serializing data':<20}")
    out = pd.DataFrame(rows, 
        columns = ["state_code", "state", "district", "sero_0", "N_0", "sero_1", "N_1", "sero_2", "N_2", "sero_3", "N_3", "sero_4", "N_4", "sero_5", "N_5", "sero_6", "N_6", "N_tot", "Rt", "S0", "I0", "R0", "D0", "dT0", "dD0"]
    )
    progress.update(1)
    return out 

if __name__ == "__main__":
    # assemble_sero_data().to_csv(data/"all_india_sero_pop.csv")
    assemble_initial_conditions(focus_states).to_csv(data/"focus_states_simulation_initial_conditions.csv")
    # assemble_initial_conditions().to_csv(data/"all_india_simulation_initial_conditions.csv")
