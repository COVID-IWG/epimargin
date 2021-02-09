from itertools import product
from pathlib import Path

import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import state_code_lookup
from adaptive.models import SIR
from adaptive.policy import PrioritizedAssignment, RandomVaccineAssignment
from studies.age_structure.common_TN_data import *

sns.set(style = "whitegrid")

data = Path("./data").resolve()

(state, date, seropos, sero_breakdown) = ("TN", "October 23, 2020", TN_seropos, TN_sero_breakdown)
N = india_pop[state_code_lookup[state].replace("&", "and")]

# scaling
dT_conf = df[state].loc[:, "delta", "confirmed"] 
dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index)
T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N * seropos)
T_ratio = T_sero/T

# grab time series 
R = df[state].loc[simulation_start, "total", "recovered"]
D = df[state].loc[simulation_start, "total", "deceased"]
S = T_sero - R - D

# run Rt estimation on scaled timeseries 
(Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_smooth, CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

immunity_threshold = 0.75

for (vax_pct_annual_goal, vax_effectiveness) in product(
    (0.25, 0.50),
    (0.70, 1.00)
):
    daily_rate = vax_pct_annual_goal/365
    daily_vax_doses = int(vax_effectiveness * daily_rate * N)
    
    if vax_pct_annual_goal == 0:
        if vax_effectiveness != 1.00: 
            continue
        policies = [RandomVaccineAssignment(daily_vax_doses, IN_age_ratios)]
    else: 
        policies = [
            RandomVaccineAssignment(daily_vax_doses, IN_age_ratios), 
            PrioritizedAssignment(daily_vax_doses, split_by_age(S), [6, 5, 4, 3, 2, 1, 0], "mortality"), 
            PrioritizedAssignment(daily_vax_doses, split_by_age(S), [1, 2, 3, 4, 0, 5, 6], "contactrate")
        ]
    
    for vaccination_policy in policies:
        model = SIR(
            name        = state, 
            population  = N, 
            dT0         = np.ones(num_sims) * (dT_conf_smooth[simulation_start] * T_ratio).astype(int), 
            Rt0         = Rt[simulation_start] * N/(N - T_sero),
            I0          = np.ones(num_sims) * S, 
            R0          = np.ones(num_sims) * R, 
            D0          = np.ones(num_sims) * D,
            random_seed = 0
        )

        t = 0
        dVx = [np.zeros(len(IN_age_structure))]

        # run vax rate forward 1 year, then until 75% of pop is recovered or vax
        while (t <= 365) or ((t > 365) and (model.R[-1].mean() + (daily_vax_doses * t))/N < immunity_threshold):
            dVx.append(vaccination_policy.distribute_doses(model))
            print("::::", state, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]))
            t += 1
            if vax_pct_annual_goal == 0 and t > 365 * 5:
                break 

        tag = f"{state}_"
        if vax_pct_annual_goal == 0:
            tag += "novaccination"
        else: 
            tag += "_".join([
                vaccination_policy.name(),
                f"ve{int(100*vax_effectiveness)}",
                f"annualgoal{int(100 * vax_pct_annual_goal)}",
                f"threshold{int(100 * immunity_threshold)}"
            ])

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
