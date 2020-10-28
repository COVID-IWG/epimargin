import adaptive.plots as plt
from adaptive.models import SIR
from adaptive.estimators import analytical_MPVS, parametric_scheme_mcmc, branching_random_walk
from adaptive.smoothing import convolution
import numpy as np 
import pandas as pd 
import pymc3 as pm 

sir_model = SIR("test", population = 100000, dT0 = 10, Rt0 = 1.1, random_seed = 0)

total_t = 0
for (Rt, t) in [(1.1, 25), (1.6, 100), (0.99, 100)]:
    sir_model.Rt0 = Rt 
    sir_model.run(t)
    total_t += t

# plt.figure()
# plt.plot(sir_model.dT)
# plt.figure()
# plt.plot(sir_model.Rt)
# plt.show()


# model, trace, summary = parametric_scheme_mcmc(sir_model.dT, chains = 4, draws = 1000)
model, trace, summary = branching_random_walk(sir_model.dT, chains = 4, draws = 1000)

pm.traceplot(trace)
plt.show()

plt.plot(range(total_t + 1), sir_model.Rt)
plt.plot(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["mean"], color = "orange")
plt.fill_between(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_2.5%"], summary.loc[[_ for _ in summary.index if _.startswith("Rt")]]["hdi_97.5%"], color = "orange", alpha = 0.3)
plt.ylim(0, 4)
plt.xlim(0, total_t)
plt.title("$R_t$ - naive MCMC")
plt.show()

# plt.plot(range(total_t + 1), sir_model.dT, color = "black")
# plt.plot(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["mean"], color = "blue")
# plt.fill_between(range(1, total_t + 1), summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["hdi_2.5%"], summary.loc[[_ for _ in summary.index if _.startswith("dT")]]["hdi_97.5%"], color = "blue", alpha = 0.3)
# plt.show()

_, Rt, Rt_lb, Rt_ub, *_, anomalies, anomaly_dates = analytical_MPVS(pd.DataFrame(sir_model.dT), smoothing = convolution("uniform", 3), CI = 0.95, totals = False)

plt.plot(range(total_t + 1), sir_model.Rt)
plt.plot(range(1, total_t), Rt, color = "orange")
plt.fill_between(range(1, total_t), Rt_ub, Rt_lb, color = "orange", alpha = 0.3)
plt.vlines(anomaly_dates, 0, 4, colors = "red", linestyles = "dotted", linewidth = 1)
plt.ylim(0, 4)
plt.xlim(0, total_t)
plt.title("$R_t$ - parametric scheme")
plt.show()