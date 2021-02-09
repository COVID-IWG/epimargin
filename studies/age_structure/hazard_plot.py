from pathlib import Path

import adaptive.plots as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import state_code_lookup
from adaptive.models import SIR
from matplotlib import dates as mdates
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
S = N - T_sero - R - D

# run Rt estimation on scaled timeseries 
(Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_smooth, CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

immunity_threshold = 0.75


model = SIR(
    name        = state, 
    population  = N, 
    dT0         = np.ones(num_sims) * (dT_conf_smooth[simulation_start] * T_ratio).astype(int), 
    Rt0         = Rt[simulation_start],
    I0          = np.ones(num_sims) * S, 
    R0          = np.ones(num_sims) * R, 
    D0          = np.ones(num_sims) * D,
    mortality   = mu_TN,
    random_seed = 0
)

# run vax rate forward 1 year, then until 75% of pop is recovered or vax
t = 0
while (t < 1 or np.mean(model.dD[-1]) > 0):
    print("::::", state, t, np.mean(model.dT[-1]), np.std(model.dT[-1]))
    model.parallel_forward_epi_step()
    t += 1

dDx = fD * [_.mean() for _ in model.dD[1:]]
Sx  = fS * [_.mean() for _ in model.S [1:]]

prob_death = dDx/Sx

sns.set_palette(sns.color_palette("hls", 7))
dt = [simulation_start + pd.Timedelta(n, "days") for n in range(t)]
PrD = pd.DataFrame(prob_death).T\
    .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    .assign(t = dt)\
    .set_index("t")


PrD.plot()
plt.legend(title = "Age category", title_fontsize = 18, fontsize = 16, framealpha = 1, handlelength = 1)
plt.xlim(right = pd.Timestamp("Feb 01, 2021"))
plt.PlotDevice()\
    .xlabel("\ndate")\
    .ylabel("probability\n")
plt.subplots_adjust(left = 0.12, bottom = 0.12, right = 0.94, top = 0.96)
plt.gca().xaxis.set_minor_locator(mpl.ticker.NullLocator())
plt.gca().xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(fontsize = "16")
plt.yticks(fontsize = "16")
plt.gca().xaxis.grid(True, which = "major")
plt.show()

# tag = f"{state}_"

# # calculate hazards and probability 
# dTx = sero_breakdown[..., None] * [_.mean().astype(int) for _ in model.dT]
# Sx  = IN_age_ratios[..., None]  * [_.mean().astype(int) for _ in model.S]
# lambda_x = dTx/Sx
# Pr_covid_t     = np.zeros(lambda_x.shape)
# Pr_covid_pre_t = np.zeros(lambda_x.shape)
# Pr_covid_t[:, 0]     = lambda_x[:, 0]
# Pr_covid_pre_t[:, 0] = lambda_x[:, 0]
# for t in range(1, len(lambda_x[0, :])):
#     Pr_covid_t[:, t] = lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])
#     Pr_covid_pre_t[:, t] = Pr_covid_pre_t[:, t-1] + lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])

#     # # save hazards, probabilities
#     # pd.DataFrame(lambda_x).T\
#     #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
#     #     .to_csv(data/f"lambdax_{tag}.csv")
    
#     # pd.DataFrame(Pr_covid_pre_t).T\
#     #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
#     #     .to_csv(data/f"Pr_cov_pre_{tag}.csv")
#     # pd.DataFrame(Pr_covid_t).T\
#     #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
#     #     .to_csv(data/f"Pr_cov_at_{tag}.csv")

#     # # save vaccine dose timeseries 
#     # pd.DataFrame(dVx)\
#     #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
#     #     .to_csv(data/f"dVx_{tag}.csv")
    
#     # # save recovery timeseries
#     # pd.DataFrame([split_by_age(_.mean()).astype(int) for _ in model.R])\
#     #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
#     #     .to_csv(data/f"Rx_{tag}.csv")

# sns.set_palette(sns.color_palette("hls", 7))
# pd.DataFrame(Pr_covid_t).T.rename(columns = dict(enumerate(IN_age_structure.keys()))).plot()
# plt.legend(title = "Age category")
# plt.xlim(left = 0, right = t)
# plt.PlotDevice()\
#     .xlabel("\ndate")\
#     .ylabel("probability\n")\
#     .title("\nProbability of contracting SARS-CoV-2 over time, by age group")\
#     .annotate("Tamil Nadu, statewide simulation, no vaccination")
# plt.show()


# per-capita cases, deaths
# date_index = pd.date_range(ts.index.get_level_values(1).min(), ts.index.get_level_values(1).max())
# percap = ts.loc[list(district_populations.keys())][["dD", "dT"]]\
#     .reset_index()
# percap["month"] = percap.status_change_date.dt.month.astype(str) + "_" + percap.status_change_date.dt.year.astype(str)
# percap["N"] = percap["detected_district"].replace(district_populations)
# percap["dD"] = percap["dD"]/percap["N"]
# percap["dT"] = percap["dT"]/percap["N"]
# percap.groupby(["detected_district", "month"]).apply(np.mean)[["dD", "dT"]].to_csv(data/"TN_percap.csv")

# nat_percap = \
# df["TT"].loc[:, "delta"].unstack()[["confirmed", "deceased"]]\
#     .reset_index()\
#     .rename(columns = {"confirmed": "dT", "deceased": "dD", "index": "date"})\
#     .assign(month = lambda _: _.date.dt.month.astype(str) + "_" + _.date.dt.year.astype(str))\
#     .groupby("month")\
#     .apply(np.mean)\
#     .drop(columns = ["month"])\
#     .sort_index()/(1.3e9)

# nat_percap.to_csv(data/"IN_percap.csv")
