import epimargin.plots as plt
from epimargin.models import SIR
from epimargin.estimators import analytical_MPVS, parametric_scheme_mcmc, branching_random_walk
from epimargin.smoothing import convolution
import numpy as np 
import pandas as pd 
import pymc3 as pm 

sir_model = SIR("test", population = 500000, I0 = 100, dT0 = 20, Rt0 = 1.01, random_seed = 0)

total_t = 0
schedule = [(1.01, 75), (1.4, 75), (0.9, 75)]
R0_timeseries = []
for (R0, t) in schedule:
    R0_timeseries += [R0] * t
    sir_model.Rt0 = R0
    sir_model.run(t)
    total_t += t

plt.plot(sir_model.dT)
plt.show()
plt.plot(R0_timeseries, "-", color = "black",      label = "$R_0$")
plt.plot(sir_model.Rt,  "-", color = "dodgerblue", label = "$R_t$")
plt.legend(framealpha = 1, handlelength = 1, loc = "best")
plt.PlotDevice().xlabel("time").ylabel("reproductive rate").adjust(left = 0.10, bottom = 0.15, right = 0.99, top = 0.99)
plt.ylim(0.5, 1.5)
plt.show()

# 1: parametric scheme:
dates, Rt, Rt_lb, Rt_ub, *_, anomalies, anomaly_dates = analytical_MPVS(pd.DataFrame(sir_model.dT), smoothing = convolution("uniform", 2), CI = 0.99, totals = False)
pd = plt.Rt(dates, Rt, Rt_ub, Rt_lb, ymin = 0.5, ymax = 2.5, CI = 0.99, yaxis_colors = False, format_dates = False, critical_threshold = False)\
    .xlabel("time")\
    .ylabel("reproductive rate")\
    .adjust(left = 0.11, bottom = 0.15, right = 0.98, top = 0.98)
plt.plot(sir_model.Rt,  "-", color = "white", linewidth = 3, zorder = 10)
sim_rt, = plt.plot(sir_model.Rt,  "-", color = "dodgerblue", linewidth = 2, zorder = 11)
anoms   = plt.vlines(anomaly_dates, 0, 4, colors = "red", linewidth = 2, alpha = 0.5)
plt.legend(
    [pd.markers["Rt"], sim_rt, anoms],
    ["Estimated $R_t$ (99% CI)", "simulated $R_t$", "anomalies"],
    **pd.legend_props
)
plt.show()

# 2: naive MCMC 
model, trace, summary = parametric_scheme_mcmc(sir_model.dT, CI = 0.99, chains = 4, draws = 1000)
Rt_pred = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["mean"][1:]
Rt_lb   = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_0.5%"][1:]
Rt_ub   = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_99.5%"][1:]
pd = plt.Rt(dates, Rt_pred, Rt_ub, Rt_lb, ymin = 0.5, ymax = 2.5, CI = 0.99, yaxis_colors = False, format_dates = False, critical_threshold = False)\
    .xlabel("time")\
    .ylabel("reproductive rate")\
    .adjust(left = 0.11, bottom = 0.15, right = 0.98, top = 0.98)
plt.plot(sir_model.Rt,  "-", color = "white", linewidth = 3, zorder = 10)
sim_rt, = plt.plot(sir_model.Rt,  "-", color = "dodgerblue", linewidth = 2, zorder = 11)
plt.legend(
    [pd.markers["Rt"], sim_rt],
    ["Estimated $R_t$ (99% CI)", "simulated $R_t$"],
    **pd.legend_props
)
plt.show()

# 3: branching parameter random walk 
model, trace, summary = branching_random_walk(sir_model.dT, CI = 0.99, chains = 4, draws = 1000)
Rt_pred = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["mean"][1:]
Rt_lb   = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_0.5%"][1:]
Rt_ub   = summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_99.5%"][1:]
pd = plt.Rt(dates, Rt_pred, Rt_ub, Rt_lb, ymin = 0.5, ymax = 2.5, CI = 0.99, yaxis_colors = False, format_dates = False, critical_threshold = False)\
    .xlabel("time")\
    .ylabel("reproductive rate")\
    .adjust(left = 0.11, bottom = 0.15, right = 0.98, top = 0.98)
plt.plot(sir_model.Rt,  "-", color = "white", linewidth = 3, zorder = 10)
sim_rt, = plt.plot(sir_model.Rt,  "-", color = "dodgerblue", linewidth = 2, zorder = 11)
plt.legend(
    [pd.markers["Rt"], sim_rt],
    ["Estimated $R_t$ (99% CI)", "simulated $R_t$"],
    **pd.legend_props
)
plt.show()
# pm.traceplot(trace)
# plt.show()

# plt.plot(range(total_t + 1), sir_model.Rt)
# plt.plot(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["mean"], color = "orange")
# plt.fill_between(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_2.5%"], summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_97.5%"], color = "orange", alpha = 0.3)
# plt.ylim(0, 4)
# plt.xlim(0, total_t)
# plt.title("$R_t$ - naive MCMC")
# plt.show()

# # plt.plot(range(total_t + 1), sir_model.dT, color = "black")
# # plt.plot(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["mean"], color = "blue")
# # plt.fill_between(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["hdi_2.5%"], summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["hdi_97.5%"], color = "blue", alpha = 0.3)
# # plt.show()

# _, Rt, Rt_lb, Rt_ub, *_, anomalies, anomaly_dates = analytical_MPVS(pd.DataFrame(sir_model.dT), smoothing = convolution("uniform", 3), CI = 0.95, totals = False)

# plt.plot(range(total_t + 1), sir_model.Rt)
# plt.plot(range(1, total_t), Rt, color = "orange")
# plt.fill_between(range(1, total_t), Rt_ub, Rt_lb, color = "orange", alpha = 0.3)
# plt.vlines(anomaly_dates, 0, 4, colors = "red", linestyles = "dotted", linewidth = 1)
# plt.ylim(0, 4)
# plt.xlim(0, total_t)
# plt.title("$R_t$ - parametric scheme")
# plt.show()
