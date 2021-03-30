import numpy as np
import pandas as pd
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *

src = mkdir(ext/f"all_india_wtp_metrics{num_sims}")

today = pd.Timestamp.now()
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
        np.median(yll[tag])
    ])
    
pd.DataFrame(rows, columns = ["state", "district", "WTP", "VSLY", "YLL"])\
    .to_csv(data/f"IDFC_{today.strftime('%b%d')}.csv")