from itertools import product
from pathlib import Path

import epimargin.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epimargin.estimators import analytical_MPVS
from epimargin.etl.covid19india import state_code_lookup
from epimargin.models import SIR
from epimargin.policy import PrioritizedAssignment, RandomVaccineAssignment
from studies.age_structure.commons import * 

# first pass: one bucket 
N_natl = sum(india_pop.values())
dT_conf = df["TT"][:, "delta", "confirmed"]
dT_conf = dT_conf.reindex(pd.date_range(dT_conf.index.min(), dT_conf.index.max()), fill_value = 0)
dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index).clip(0).astype(int)
T_scaled = dT_conf_smooth.cumsum()[simulation_start] * T_ratio

D, R = df.loc[simulation_start, "TT"]["total"][["deceased", "recovered"]]
S = N_natl - T_scaled - D - R

(Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_smooth, CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

model = SIR(
    name        = state, 
    population  = N_natl, 
    dT0         = np.ones(num_sims) * (dT_conf_smooth[simulation_start] * T_ratio).astype(int), 
    Rt0         = Rt[simulation_start],
    I0          = np.ones(num_sims) * (T_scaled - R - D), 
    R0          = np.ones(num_sims) * R, 
    D0          = np.ones(num_sims) * D,
    mortality   = mu_TN,
    random_seed = 0
)

while np.mean(model.dT[-1]) > 0:
    model.parallel_forward_epi_step()

pd.DataFrame(data = { 
        "date": [simulation_start + pd.Timedelta(n, "days") for n in range(len(model.dT))],
        "dT"  : [np.mean(_)/N_natl for _ in model.dT],
        "dD"  : [np.mean(_)/N_natl for _ in model.dD],
    })\
    .assign(month = lambda _: _.date.dt.month.astype(str) + "_" + _.date.dt.year.astype(str))\
    .groupby("month")\
    .apply(np.mean)\
    .drop(columns = ["month"])\
    .to_csv(data/"IN_simulated_percap.csv")

"""
- new sims in dropbox
- deaths and YLL figures on overleaf
- looking at econ stuff now 

questions: 
- do we really want to do a national model that is a superposition of 730 district models? 
- do we want backtesting figures?
- 
"""
