import adaptive.plots as plt
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import state_code_lookup
from studies.age_structure.commons import *
from studies.age_structure.palette import *

import seaborn as sns
sns.set(style = "white")
# state
TN_N = india_pop[state_code_lookup[state].replace("&", "and")]
TN_dT_conf = df["TN"].loc[:, "delta", "confirmed"] 
TN_dT_conf_smooth = pd.Series(smooth(TN_dT_conf), index = TN_dT_conf.index)
TN_T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
TN_T = TN_T_conf_smooth[date]
TN_T_sero = (N * TN_seropos)
T_ratio = TN_T_sero/TN_T


# national
IN_dT_conf = df["TT"].loc[:, "delta", "confirmed"] 
IN_dT_conf_smooth = pd.Series(smooth(IN_dT_conf), index = IN_dT_conf.index)
IN_T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
IN_T = IN_T_conf_smooth[date]

# run Rt estimation on scaled timeseries 
(TN_dates, TN_Rt_est, TN_CI_l, TN_CI_u, *_) =\
    analytical_MPVS(T_ratio * TN_dT_conf_smooth["March 1, 2020":simulation_start], CI = CI, smoothing = lambda _:_, totals = False)

(IN_dates, IN_Rt_est, IN_CI_l, IN_CI_u, *_) =\
    analytical_MPVS(T_ratio * IN_dT_conf_smooth["March 1, 2020":simulation_start], CI = CI, smoothing = lambda _:_, totals = False)


TN_CI_marker  = plt.fill_between(TN_dates, TN_CI_l, TN_CI_u, color = TN_color, alpha = 0.5, zorder = 5)
TN_Rt_marker, = plt.plot(TN_dates, TN_Rt_est, color = TN_color, linewidth = 2, zorder = 5, solid_capstyle = "butt")
plt.plot(TN_dates, TN_CI_l, color = TN_color, linewidth = 0.5, zorder = 5, solid_capstyle = "butt")
plt.plot(TN_dates, TN_CI_u, color = TN_color, linewidth = 0.5, zorder = 5, solid_capstyle = "butt")

IN_CI_marker  = plt.fill_between(IN_dates, IN_CI_l, IN_CI_u, color = plt.BLK, alpha = 0.5, zorder = 10)
IN_Rt_marker, = plt.plot(IN_dates, IN_Rt_est, color = plt.BLK, linewidth = 2, zorder = 10, solid_capstyle = "butt")
plt.plot(IN_dates, IN_CI_l, color = plt.BLK, linewidth = 0.5, zorder = 10, solid_capstyle = "butt")
plt.plot(IN_dates, IN_CI_u, color = plt.BLK, linewidth = 0.5, zorder = 10, solid_capstyle = "butt")

plt.xticks(fontsize = "20", rotation = 45)
plt.yticks(fontsize = "20")
plt.legend(
    [(TN_CI_marker, TN_Rt_marker), (IN_CI_marker, IN_Rt_marker)], 
    [f"Tamil Nadu", "India"], 
    framealpha = 1, handlelength = 0.75, fontsize = "20", 
    loc = "lower center", bbox_to_anchor = (0.5, 1),
    ncol = 2
)
plt.xlim(left = pd.Timestamp("March 1, 2020"), right = simulation_start)
plt.ylim(bottom = 0.5, top = 3.25)
plt.gca().xaxis.set_major_formatter(plt.DATE_FMT)
plt.gca().xaxis.set_minor_formatter(plt.DATE_FMT)
plt.PlotDevice().ylabel("reproductive rate\n")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()


dT_slice = dT_conf_smooth["March 1, 2020":simulation_start]
dT_idx = dT_slice.index
fig = plt.figure()
tn_scatter = plt.scatter(dT_slice.index, TN_dT_conf["March 1, 2020":simulation_start]        * T_ratio/TN_N, color = TN_color, label = "Tamil Nadu (raw)",      figure = fig, alpha = 0.5, marker = "o", s = 10, zorder = 5)
tn_plot,   = plt.plot   (dT_slice.index, TN_dT_conf_smooth["March 1, 2020":simulation_start] * T_ratio/TN_N, color = TN_color, label = "Tamil Nadu (smoothed)", figure = fig, zorder = 5)
in_scatter = plt.scatter(dT_slice.index, IN_dT_conf["March 1, 2020":simulation_start]        * T_ratio/(1.389e9), color = IN_color, label = "India (raw)",      figure = fig, alpha = 0.5, marker = "o", s = 10, zorder = 10)
in_plot,   = plt.plot   (dT_slice.index, IN_dT_conf_smooth["March 1, 2020":simulation_start] * T_ratio/(1.389e9), color = IN_color, label = "India (smoothed)", figure = fig, zorder = 10)
plt.xticks(fontsize = "20", rotation = 45)
plt.yticks(fontsize = "20")
plt.legend(
    [tn_scatter, tn_plot, in_scatter, in_plot], 
    ["Tamil Nadu (raw)", "Tamil Nadu (smoothed)", "India (raw)", "India (smoothed)"],
    fontsize = "20", ncol = 4,     
    framealpha = 1, handlelength = 0.75,
    loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.gca().xaxis.set_major_formatter(plt.DATE_FMT)
plt.gca().xaxis.set_minor_formatter(plt.DATE_FMT)
plt.xlim(left = pd.Timestamp("March 1, 2020"), right = simulation_start)
plt.PlotDevice().ylabel("per-capita infection rate\n")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
