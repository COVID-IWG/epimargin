from itertools import product

import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import SIR
from scipy.stats import multinomial as Multinomial

from .common_TN_data import * 

sns.set(style = "whitegrid")

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

    geo_tag       = f"{state}_{district}_randomassignment_"
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
