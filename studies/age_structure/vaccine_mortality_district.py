from itertools import product
from pathlib import Path

import adaptive.plots as plt
import flat_table
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import (data_path, get_time_series,
                                       load_all_data, state_code_lookup)
from adaptive.models import SIR
from adaptive.smoothing import notched_smoothing
from scipy.stats import multinomial as Multinomial

sns.set(style = "whitegrid")

data = Path("./data").resolve()


# Rt estimation parameters
CI = 0.95
window = 14
gamma = 0.2
infectious_period = 5
smooth = notched_smoothing(window)

# simulation parameters
simulation_start = pd.Timestamp("Jan 1, 2021")
num_sims = 10000

# common vaccination parameters
immunity_threshold = 0.75 


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
    # 0:  116_880,
    # 5:  117_982 + 126_156 + 126_046,
    # 18: 122_505 + 117_397, 
    # 30: 112_176 + 103_460,
    # 40: 90_220 + 79_440,
    # 50: 68_876 + 59_256 + 48_891,
    # 65: 38_260 + 24_091,
    # 75: 15_084 + 8_489 + 3_531 +  993 +  223 +  48,
}

district_populations = { 
    'Ariyalur'       :   754_894, # 'Ariyalur'
    'Chengalpattu'   : 2_556_244, # 'Chengalpattu'
    'Chennai'        : 4_646_732, # 'Chennai'
    'Coimbatore'     : 3_458_045, # 'Coimbatore'
    'Cuddalore'      : 2_605_914, # 'Cuddalore'
    'Dharmapuri'     : 1_506_843, # 'Dharmapuri'
    'Dindigul'       : 2_159_775, # 'Dindigul'
    'Erode'          : 2_251_744, # 'Erode'
    'Kallakurichi'   : 1_370_281, # 'Kallakurichi'
    'Kancheepuram'   : 1_166_401, # 'Kanchipuram'
    'Kanyakumari'    : 1_870_374, # 'Kanniyakumari'
    'Karur'          : 1_064_493, # 'Karur'
    'Krishnagiri'    : 1_879_809, # 'Krishnagiri'
    'Madurai'        : 3_038_252, # 'Madurai'
    # 'Mayiladuthurai' :   918_356, # 'Mayiladuthurai'
    'Nagapattinam'   :   697_069, # 'Nagapattinam'
    'Namakkal'       : 1_726_601, # 'Namakkal'
    'Nilgiris'       :   735_394, # 'Nilgiris'
    'Perambalur'     :   565_223, # 'Perambalur'
    'Pudukkottai'    : 1_618_345, # 'Pudukkottai'
    'Ramanathapuram' : 1_353_445, # 'Ramanathapuram'
    'Ranipet'        : 1_210_277, # 'Ranipet'
    'Salem'          : 3_482_056, # 'Salem'
    'Sivaganga'      : 1_339_101, # 'Sivagangai'
    'Tenkasi'        : 1_407_627, # 'Tenkasi'
    'Thanjavur'      : 2_405_890, # 'Thanjavur'
    'Theni'          : 1_245_899, # 'Theni'
    'Thiruvallur'    : 3_728_104, # 'Tiruvallur'
    'Thiruvarur'     : 1_264_277, # 'Tiruvarur'
    'Thoothukkudi'   : 1_750_176, # 'Thoothukudi'
    'Tiruchirappalli': 2_722_290, # 'Tiruchirappalli'
    'Tirunelveli'    : 1_665_253, # 'Tirunelveli'
    'Tirupathur'     : 1_111_812, # 'Tirupattur'
    'Tiruppur'       : 2_479_052, # 'Tiruppur'
    'Tiruvannamalai' : 2_464_875, # 'Tiruvannamalai'
    'Vellore'        : 1_614_242, # 'Vellore'
    'Viluppuram'     : 2_093_003, # 'Viluppuram'
    'Virudhunagar'   : 1_942_288, # 'Virudhunagar'
}

# normalize
age_structure_norm = sum(IN_age_structure.values())
IN_age_ratios = np.array([v/age_structure_norm for v in IN_age_structure.values()])
split_by_age = lambda v: (v * IN_age_ratios).astype(int)

# from Karnataka
COVID_age_ratios = np.array([0.01618736, 0.07107746, 0.23314877, 0.22946212, 0.18180406, 0.1882451 , 0.05852026, 0.02155489])

india_pop = pd.read_csv(data/"india_pop.csv", names = ["state", "population"], index_col = "state").to_dict()["population"]

# load covid19 india data 
print(":: loading case timeseries data")
# download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)

# download district-level data 
paths = {"v3": [data_path(i) for i in (1, 2)], "v4": [data_path(i) for i in range(3, 22)]}
# for target in paths['v3'] + paths['v4']: download_data(data, target)
ts = load_all_data(v3_paths = [data/filepath for filepath in paths['v3']],  v4_paths = [data/filepath for filepath in paths['v4']])\
    .query("detected_state == 'Tamil Nadu'")\
    .pipe(lambda _: get_time_series(_, "detected_district"))\
    .drop(columns = ["date", "time", "delta", "logdelta"])\
    .rename(columns = {
        "Deceased":     "dD",
        "Hospitalized": "dT",
        "Recovered":    "dR"
    })

print(":: seroprevalence scaling")
TN_sero_breakdown = np.array([0.311, 0.311, 0.320, 0.333, 0.320, 0.272, 0.253]) # from TN sero, assume 0-18 sero = 18-30 sero
TN_pop = india_pop["Tamil Nadu"]
TN_seropos = split_by_age(TN_pop) @ TN_sero_breakdown/TN_pop

(state, date, seropos, sero_breakdown) = ("TN", "October 23, 2020", TN_seropos, TN_sero_breakdown)
N = india_pop[state_code_lookup[state].replace("&", "and")]

# scaling
dT_conf = df[state].loc[:, "delta", "confirmed"] 
dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index)
T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N * seropos)
T_ratio = T_sero/T

print(":: running simulations")
for ((district, N_district), vax_pct_annual_goal, vax_effectiveness) in product(
    district_populations.items(),
    (0, 0.25, 0.50),
    (0.70, 1.00)
):
    if vax_pct_annual_goal == 0 and vax_effectiveness != 1.00:
        continue
    # grab time series 
    D, R = ts.loc[district][["dD", "dR"]].sum()

    dT_conf_district = ts.loc[district].dT
    dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
    dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)

    # run Rt estimation on scaled timeseries 
    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    daily_rate = vax_pct_annual_goal/365
    daily_vax_doses = int(vax_effectiveness * daily_rate * N_district)

    T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio

    model = SIR(
        name        = state, 
        population  = N_district, 
        dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
        Rt0         = Rt[simulation_start] * N_district/(N_district - T_scaled),
        I0          = np.ones(num_sims) * (T_scaled - R - D), 
        R0          = np.ones(num_sims) * R, 
        D0          = np.ones(num_sims) * D,
        random_seed = 0
    )

    t = 0
    dVx = [np.zeros(len(IN_age_structure))]

    # run vax rate forward 1 year, then until 75% of pop is recovered or vax
    while (t <= 365) or ((t > 365) and (model.R[-1].mean() + (daily_vax_doses * t))/N_district < immunity_threshold):
        dVx.append(Multinomial.rvs(daily_vax_doses, IN_age_ratios))
        model.S[-1] -= daily_vax_doses
        model.parallel_forward_epi_step()
        print("::::", state, district, vax_pct_annual_goal, vax_effectiveness, t, np.mean(model.dT[-1]), np.std(model.dT[-1]))
        t += 1
        if vax_pct_annual_goal == 0 and t > 365 * 5:
            break 

    geo_tag       = f"{state}_{district}_mortalityprioritized_"
    parameter_tag = "novaccination" if vax_pct_annual_goal == 0 else f"ve{int(100*vax_effectiveness)}_annualgoal{int(100 * vax_pct_annual_goal)}_threshold{int(100 * immunity_threshold)}"
    tag = geo_tag + parameter_tag
    
    print(":::: serializing results")
    
    # calculate hazards and probability 
    dTx = sero_breakdown[..., None] * [_.mean().astype(int) for _ in model.dT]
    Sx  = IN_age_ratios[..., None]  * [_.mean().astype(int) for _ in model.S]
    lambda_x = dTx/Sx
    Pr_covid_t     = np.zeros(lambda_x.shape)
    Pr_covid_pre_t = np.zeros(lambda_x.shape)
    Pr_covid_t[:, 0]     = lambda_x[:, 0]
    Pr_covid_pre_t[:, 0] = lambda_x[:, 0]
    for t in range(1, len(lambda_x[0, :])):
        Pr_covid_t[:, t] = lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])
        Pr_covid_pre_t[:, t] = Pr_covid_pre_t[:, t-1] + lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])

    # save hazards, probabilities
    pd.DataFrame(lambda_x).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"lambdax_{tag}.csv")
    
    pd.DataFrame(Pr_covid_pre_t).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Pr_cov_pre_{tag}.csv")
    pd.DataFrame(Pr_covid_t).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Pr_cov_at_{tag}.csv")

    # save vaccine dose timeseries 
    pd.DataFrame(dVx)\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"dVx_{tag}.csv")
    
    # save recovery timeseries
    pd.DataFrame([split_by_age(_.mean()).astype(int) for _ in model.R])\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Rx_{tag}.csv")
