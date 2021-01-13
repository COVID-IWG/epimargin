from pathlib import Path

import adaptive.plots as plt
import numpy as np
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data, state_code_lookup
from adaptive.models import SIR
from adaptive.smoothing import notched_smoothing
import flat_table

# root = Path(__file__).parent
# data = root/"data"
data = Path("./data").resolve()

CI = 0.95
window = 10
gamma = 0.2
infectious_period = 5
num_sims = 10000

# load admin data on population
IN_age_structure = { # WPP2019_POP_F01_1_POPULATION_BY_AGE_BOTH_SEXES
    0:  116_880,
    5:  117_982 + 126_156 + 126_046,
    18: 122_505 + 117_397, 
    30: 112_176 + 103_460,
    40: 90_220 + 79_440,
    50: 68_876 + 59_256 + 48_891,
    65: 38_260 + 24_091,
    75: 15_084 + 8_489 + 3_531 +  993 +  223 +  48,
}
# normalize
age_structure_norm = sum(IN_age_structure.values())
IN_age_ratios = np.array([v/age_structure_norm for (k, v) in IN_age_structure.items()])
split_by_age = lambda v: (v * IN_age_ratios).astype(int)

# from Karnataka
COVID_age_ratios = np.array([0.01618736, 0.07107746, 0.23314877, 0.22946212, 0.18180406, 0.1882451 , 0.05852026, 0.02155489])


india_pop = pd.read_csv(data/"india_pop.csv", names = ["state", "population"], index_col = "state").to_dict()["population"]
india_pop["Odisha"]      = india_pop["Orissa"]
india_pop["Puducherry"]  = india_pop["Pondicherry"]
india_pop["Uttarakhand"] = india_pop["Uttaranchal"]

# load covid19 india data 
download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)

# load Rt data 
Rt = pd.read_csv("data/Rt_timeseries_india.csv") 

date = "2020-12-24"

for state in set(df.loc[date, :, :].columns) - {"TT", "LA", "SK", "NL"}: 
    N = india_pop[state_code_lookup[state].replace("&", "and")]
    T = df[state].loc[date, "total", "confirmed"]
    R = df[state].loc[date, "total", "recovered"]
    D = df[state].loc[date, "total", "deceased"]
    model = SIR(
        name        = state, 
        population  = N - D, 
        dT0         = np.ones(num_sims) * df[state].loc[date, "delta", "confirmed"], 
        Rt0         = Rt[(Rt.state == state) & (Rt.date == date)].Rt.iloc[0], 
        I0          = np.ones(num_sims) * (T - R - D), 
        R0          = np.ones(num_sims) * R, 
        D0          = np.ones(num_sims) * D,
        random_seed = 0
    )
    i = 0
    while np.mean(model.dT[-1]) > 0:
        model.parallel_forward_epi_step(num_sims = num_sims)
        i += 1
        print(state, i, np.mean(model.dT[-1]), np.std(model.dT[-1]))
    dT  = np.array([_.mean().astype(int) for _ in model.dT])
    dTx = (dT * COVID_age_ratios[..., None]).astype(int)
    Tx  = (T * COVID_age_ratios).astype(int)[..., None] + dTx.cumsum(axis = 1)
    Nx  = split_by_age(N)
    lambda_x = dT/(Nx[..., None] - Tx)
    pd.DataFrame(lambda_x).to_csv(data/f"{state}_age_hazards.csv")
    pd.DataFrame(model.dT).T.to_csv(data/f"{state}_sims.csv")