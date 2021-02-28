import adaptive.plots as plt
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import state_code_lookup
from studies.age_structure.commons import *
from studies.age_structure.palette import *

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

(Rt_dates, Rt_est, CI_l, CI_u, *_) = analytical_MPVS(T_ratio * dT_conf_smooth["Feb 29, 2020":simulation_start], CI = CI, smoothing = lambda _:_, totals = False)
Rt = dict(zip(Rt_dates, Rt_est))

plt.Rt(Rt_dates, Rt_est, CI_l, CI_u, 0.95, ymin = 0.5, ymax = 3.5).show()

dT_slice = dT_conf_smooth["Feb 29, 2020":simulation_start]
dT_idx = dT_slice.index
fig = plt.figure()
plt.scatter(dT_slice.index, dT_conf["Feb 29, 2020":simulation_start]        * T_ratio/N, color = TN_color, label = "seroprevelance-scaled infection rate",        figure = fig, alpha = 0.5, marker = "o", s = 10)
plt.plot   (dT_slice.index, dT_conf_smooth["Feb 29, 2020":simulation_start] * T_ratio/N, color = TN_color, label = "smoothed, scaled infection rate time series", figure = fig)
plt.legend()
plt.show()
