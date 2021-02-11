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
from studies.age_structure.commons import *

sns.set(style = "whitegrid")
data = Path("./data").resolve()

(state, date, seropos, sero_breakdown) = ("TN", "October 23, 2020", TN_seropos, TN_sero_breakdown)
N = india_pop[state_code_lookup[state].replace("&", "and")]

models = {}

max_t = 0

for (district, N_district) in district_populations.items():
    # grab timeseries 
    D, R = ts.loc[district][["dD", "dR"]].sum()

    dT_conf_district = ts.loc[district].dT
    dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
    dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
    T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
    S = N_district - T_scaled

    # run Rt estimation on scaled timeseries 
    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    geo_tag = f"{state}_{district}_"

    # # run model forward with no vaccination
    model = SIR(
        name        = district, 
        population  = N_district, 
        dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
        Rt0         = Rt[simulation_start] * N_district/(N_district - T_scaled),
        I0          = np.ones(num_sims) * (T_scaled - R - D), 
        R0          = np.ones(num_sims) * R, 
        D0          = np.ones(num_sims) * D,
        random_seed = 0
    )

    t = 0
    while (t < 1 or np.mean(model.dT[-1]) > 0):
        print("::::", district, t, np.mean(model.dT[-1]), np.std(model.dT[-1]))
        model.parallel_forward_epi_step()
        t += 1
    max_t = max(max_t, t)

    models[district] = model 

dDx = fD * sum(np.pad([np.mean(_) for _ in model.dD[1:]], (0, max_t - len(model.Rt))) for model in models.values())
Sx  = fS * sum(np.pad([np.mean(_) for _ in model.S [1:]], (0, max_t - len(model.Rt))) for model in models.values())

all_dT = sum(np.pad([np.mean(_) for _ in model.dT[1:]], (0, max_t - len(model.Rt))) for model in models.values())

prob_death = dDx/Sx

sns.set_palette(sns.color_palette("hls", 7))
dt = [simulation_start + pd.Timedelta(n, "days") for n in range(max_t-1)]
PrD = pd.DataFrame(prob_death).T\
    .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    .assign(t = dt)\
    .set_index("t")
PrD.plot()
plt.legend(title = "Age category", title_fontsize = 18, fontsize = 16, framealpha = 1, handlelength = 1)
plt.xlim(right = pd.Timestamp("Feb 01, 2021"))
plt.PlotDevice()\
    .xlabel("\nDate")\
    .ylabel("Probability\n")
plt.subplots_adjust(left = 0.12, bottom = 0.12, right = 0.94, top = 0.96)
plt.gca().xaxis.set_minor_locator(mpl.ticker.NullLocator())
plt.gca().xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(fontsize = "16")
plt.yticks(fontsize = "16")
plt.gca().xaxis.grid(True, which = "major")
plt.semilogy()
plt.ylim(bottom = 1e-7)
plt.show()