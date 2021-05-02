import numpy as np
import pandas as pd
from studies.vaccine_allocation.commons import *
from studies.vaccine_allocation.epi_simulations import *

num_sims = 100
src = mkdir(ext/f"all_india_wtp_metrics{num_sims}")

# today = pd.Timestamp.now()
today = pd.Timestamp("March 28, 2021")
idx   = 1 + (today - simulation_start).days

wtp  = np.load(src/'district_WTP.npz')
yll  = np.load(src/'district_YLL.npz')
vsly = np.load(src/'district_VSLY.npz')

rows = []
for tag in [_ for _ in wtp.files if _.endswith("50_random")]:
    state, district, *_ = tag.split("_")
    *age_pop, N_tot = districts_to_run.loc[state, district].filter(like = "N_", axis = 0)
    pop_weight = np.array(age_pop)/N_tot
    rows.append([
        state, district, 
        np.median(wtp [tag][idx], axis = 0) @ pop_weight, 
        np.median(vsly[tag][idx], axis = 0) @ pop_weight, 
        np.median(yll[tag])/(districts_to_run.loc[state, district].N_tot/(1e6))
    ])
    
pd.DataFrame(rows, columns = ["state", "district", "WTP", "VSLY", "YLL_per_million"])\
    .to_csv(data/f"IDFC_{today.strftime('%b%d')}.csv")
