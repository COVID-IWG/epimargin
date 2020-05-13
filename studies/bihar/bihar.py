from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import etl
from adaptive.estimators import rollingOLS
from adaptive.model import Model, ModelUnit
from adaptive.plotting import plot_curve, gantt_chart
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks

seed  = 25
gamma = 0.2
window = 2


def gantt_chart(gantt_data, title, seed, show_cbar = True, filename: Optional[Path] = None):
    gantt_df = pd.DataFrame(gantt_data, columns = ["district", "day", "beta", "R"])
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    start_timestamp = pd.to_datetime("March  17, 2020")
    xlabels = [start_timestamp + pd.Timedelta(days = n) for n in gantt_df.day.unique()]
    ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        annot = gantt_pv["R"], annot_kws={"size": 8},
        cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"],
        cbar = show_cbar,
        yticklabels = gantt_df["district"].unique(),
        xticklabels=[str(xl.day) + " " + xl.month_name()[:3] for xl in xlabels],
        cbar_kws = {
            "ticks":[0.5, 1, 2, 2.5], 
            "label": "Mobility", 
            "format": mlp.ticker.FuncFormatter(lambda x, pos: {0.5:"voluntary", 1:"cautionary", 2:"partial", 2.5:"restricted"}[x]), 
            "orientation": "horizontal", 
            "aspect": 50, 
            "drawedges": True,
            "fraction": 0.05,
            "pad": 0.10, 
            "shrink": 0.5
        }
    )
    plt.xlabel("Date", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
    plt.ylabel(None)
    plt.title(title, {"size": 20, "family": "Fira Sans", "fontweight": "500"}, loc="left")
    plt.annotate(
        f"stochastic parameter: ({seed}), infectious period: {1/gamma} days, smoothing window: {window}", (0.05, 0.05), xycoords = "figure fraction", size= 8)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.10, bottom=0.10)
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")

def log_delta(ts):
    ld = pd.DataFrame(np.log(ts.cases.diff())).rename(columns = {"cases" : "logdelta"})
    ld["time"] = (ld.index - ld.index.min()).days
    return ld   

def model(districts, populations, cases, seed) -> Model:
    units = [
        ModelUnit(district, populations[i], 
        I0 = cases[district].iloc[-1].Hospitalized if district in cases.keys() else 0, 
        R0 = cases[district].iloc[-1].Recovered    if district in cases.keys() else 0, 
        D0 = cases[district].iloc[-1].Deceased     if district in cases.keys() else 0)
        for (i, district) in enumerate(districts)
    ]
    return Model(units, random_seed=seed)

def run_policies(
        district_cases: Dict[str, pd.DataFrame], # timeseries for each district 
        populations:    pd.Series,               # population for each district
        districts:      Sequence[str],           # list of district names 
        migrations:     np.matrix,               # O->D migration matrix, normalized
        gamma:          float,                   # 1/infectious period 
        Rmw:            Dict[str, float],        # mandatory regime R
        Rvw:            Dict[str, float],        # voluntary regime R
        total:          int   = 90*days,         # how long to run simulation
        eval_period:    int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling:   float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:           int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 8 day lockdown 
    model_A = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_A, 9*days, total, Rmw, Rvw, lockdown, migrations)

    # 8 day + 4 week lockdown 
    model_B = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_B, 9*days + 2*weeks, total, Rmw, Rvw, lockdown, migrations)

    # 8 day lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed).set_parameters(RR0 = Rmw)
    simulate_adaptive_control(model_C, 9*days, total, lockdown, migrations, None,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

def estimate(district, ts):
    try:
        xs = rollingOLS(etl.log_delta(ts), window = window, infectious_period = 1/gamma).R.iloc[-3:-1]
        return next((x for x in xs if not np.isnan(x) and x > 0), 1.5)
    except (ValueError, IndexError):
        return 1.5

def estimate(district, ts):
    try:
        return rollingOLS(etl.log_delta(ts), window = window, infectious_period = 1/gamma).iloc[-2:-1]
        # return next((x for x in xs if not np.isnan(x) and x > 0), 1.5)
    except (ValueError, IndexError):
        return -1 

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    sns.set(style="whitegrid", palette="bright", font="Fira Code")

    state_cases    = etl.get_time_series(etl.load_cases(data/"Bihar_Case_data_May11.csv"))
    district_cases = etl.load_cases_by_district(data/"Bihar_Case_data_May11.csv")
    district_ts    = {district: etl.get_time_series(cases) for (district, cases) in district_cases.items()}
    R_mandatory    = {district: estimate(district, ts) for (district, ts) in district_ts.items()}
    R_mandatory["JAMUI"] = 1.5 
    R_voluntary    = {district: 1.5*R for (district, R) in R_mandatory.items()}

    districts, pops, migrations = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
    replacements = {
        "PASHCHIM CHAMPARAN": "WEST CHAMPARAN", 
        "PURBA CHAMPARAN"   : "EAST CHAMPARAN", 
        "KAIMUR (BHABUA)"   : "KAIMUR", 
        "MUZAFFARPUR"       : "MUZZAFARPUR", 
        "SHEIKHPURA"        : "SHEIKPURA",
        "PURNIA"            : "PURNEA"}
    districts = [replacements.get(d.upper(), d.upper()) for d in districts]

    # run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary)

    si, sf = 0, 1000
    num_sims = sf - si

    simulation_results = [ 
        run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, seed = seed)
        for seed in tqdm(range(si, sf))
    ]

    I_100 = [tuple(m.aggregate("I") for m in ms) for ms in simulation_results]

    Ia, Ib, Ic = zip(*I_100)

    Ia_min = []
    Ia_max = []
    Ia_mdn = []

    Ib_min = []
    Ib_max = []
    Ib_mdn = []

    Ic_min = []
    Ic_max = []
    Ic_mdn = []
    for t in range(90):
        Ia_sorted = sorted([ts[t] for ts in Ia if t < len(ts)])
        Ia_min.append(Ia_sorted[0])
        Ia_max.append(Ia_sorted[-1])
        Ia_mdn.append(Ia_sorted[num_sims//2])

        Ib_sorted = sorted([ts[t] for ts in Ib if t < len(ts)])
        Ib_min.append(Ib_sorted[0])
        Ib_max.append(Ib_sorted[-1])
        Ib_mdn.append(Ib_sorted[num_sims//2])
        
        Ic_sorted = sorted([ts[t] for ts in Ic if t < len(ts)])
        Ic_min.append(Ic_sorted[0])
        Ic_max.append(Ic_sorted[-1])
        Ic_mdn.append(Ic_sorted[num_sims//2])

    # fig = plt.figure()
    state_cases = state_cases["April 01, 2020":]
    th = state_cases.index

    xp = [th.max() + pd.Timedelta(days = n) for n in range(90)]
    state_cases.Hospitalized.iloc[-1] = Ic_mdn[0]
    plt.semilogy(th, state_cases.Hospitalized, 'k-', label = "Empirical Case Data", linewidth = 2)

    plt.semilogy(xp, Ia_mdn, label = "17 May Release", linewidth = 2)
    plt.fill_between(xp, Ia_min, Ia_max, alpha = 0.2)
    
    plt.semilogy(xp, Ib_mdn, label = "31 May Release", linewidth = 2)
    plt.fill_between(xp, Ib_min, Ib_max, alpha = 0.2)

    plt.semilogy(xp, Ic_mdn, label = "Adaptive Control", linewidth = 2)
    plt.fill_between(xp, Ic_min, Ic_max, alpha = 0.2)
    
    plt.xlabel("Date", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
    plt.ylabel("Number of Infections", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
    
    plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
    plt.suptitle("")
    plt.title("Bihar Policy Scenarios: Projected Infections over Time", {"size": 36, "family": "Fira Sans", "fontweight": "500"}, loc = "left")
    plt.annotate(
        f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}", (0.05, 0.05), xycoords = "figure fraction", size= 8)
    plt.legend()
    plt.show()

    best_index = 1
    worst_index = 665

    plt.figure()
    gantt_chart(simulation_results[best_index][-1].gantt, "Bihar: Example Adaptive Lockdown Mobility Regime Scenario", best_index)
    plt.figure()
    gantt_chart(simulation_results[worst_index][-1].gantt, "Bihar: Example Adaptive Lockdown Mobility Regime Scenario 2", worst_index)
    plt.show()


    def gantt_seed(seed):
        _, _, mc = run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, seed = seed) 
        gantt_chart(mc.gantt, f"Bihar: Example Adaptive Lockdown Mobility Regime Scenario 2", seed)
        plt.gcf().set_size_inches(13, 8)
        plt.show()

    # projections
    def estimate(district, ts, window = 2):
        try:
            return rollingOLS(etl.log_delta(ts), window = window, infectious_period = 1/gamma)[["R", "Intercept", "gradient", "gradient_stderr"]]
        except (ValueError, IndexError):
            return -1 
    
    def project(p: pd.Series):
        t = (p.R - p.Intercept)/p.gradient
        return (p.R, p.Intercept + p.gradient*(t + 7), p.Intercept + p.gradient*(t + 14), np.sqrt(p.gradient_stderr))

    estimates_2 = {district: estimate(district, ts, 2) for (district, ts) in district_ts.items()}
    estimates_3 = {district: estimate(district, ts, 3) for (district, ts) in district_ts.items()}

    index_2 = {k: v.last_valid_index() if v is not -1 else v for (k, v) in estimates_2.items()}
    index_3 = {k: v.last_valid_index() if v is not -1 else v for (k, v) in estimates_3.items()}

    {district: project(est.loc[index_2[district]]) if index_2[district] is not -1 else None for (district, est) in estimates_2.items()}

    # {district: project(estimates_2[district].loc[idx]) if idx is not -1 or idx is not None else (None, None, None, None) for (district, idx) in index_2.items()}
    projections = []
    for district, estimate in estimates_2.items():
        if estimate is -1:
            projections.append((district, None, None, None, None))
        else:
            idx = index_2[district]
            if idx is None or idx is -1:
                projections.append((district, None, None, None, None))
            else: 
                projections.append((district, *project(estimate.loc[idx])))
    projdf = pd.DataFrame(data = projections, columns = ["district", "current R", "1 week projection", "2 week projection", "stderr"])