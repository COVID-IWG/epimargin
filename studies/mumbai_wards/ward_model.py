from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import etl
from adaptive.estimators import rollingOLS
from adaptive.model import Model, ModelUnit
from adaptive.plots import plot_curve
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks

seed  = 25
gamma = 0.2
Rv_Rm = 1.4836370631808469

def log_delta(ts):
    ld = pd.DataFrame(np.log(ts.cases.diff())).rename(columns = {"cases" : "logdelta"})
    ld["time"] = (ld.index - ld.index.min()).days
    return ld   

def model(wards, populations, cases, seed) -> Model:
    units = [
        ModelUnit(ward, populations.iloc[i], I0 = cases[ward].iloc[-1].cases) 
        for (i, ward) in enumerate(wards)
    ]
    return Model(units, random_seed=seed)

def get_R(ward_cases: Dict[str, pd.DataFrame], gamma: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    tsw = {ward: log_delta(cases) for (ward, cases) in ward_cases.items()}
    grw = {ward: rollingOLS(ts, infectious_period = 1/gamma) for (ward, ts) in tsw.items()}
    
    Rmw = {ward: np.mean(growth_rates.R) for (ward, growth_rates) in grw.items()}
    Rvw = {ward: Rv_Rm*Rm for (ward, Rm) in Rmw.items()}

    return Rmw, Rvw

def plot_historical(all_cases: pd.DataFrame, models: Sequence[Model], labels: Sequence[str], title, xlabel, ylabel, subtitle = None, curve: str = "I", filename = None):
    fig = plt.figure()
    th = all_cases[all_cases.ward == "ALL"].index
    plt.semilogy(th, all_cases["cases"][all_cases.ward == "ALL"], 'k-', label = "Empirical Case Data", figure = fig, linewidth = 3)
    
    for (model, label) in zip(models, labels):
        curve_data = model.aggregate(curve)
        xp = [th.max() + pd.Timedelta(days = n) for n in range(len(curve_data))]
        plt.semilogy(xp, curve_data, label = label, figure = fig, linewidth = 3)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if subtitle:
        plt.title(subtitle)
    plt.legend() 
    plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
    fig.autofmt_xdate()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi = 600)
    return fig

def gantt_chart(gantt_data, start_date, title, subtitle = None, filename: Optional[Path] = None):
    gantt_df = pd.DataFrame(gantt_data, columns = ["district", "day", "beta", "R"])
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    start_timestamp = pd.to_datetime(start_date)
    xlabels = [start_timestamp + pd.Timedelta(days = n) for n in gantt_df.day.unique()]
    ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        annot = gantt_pv["R"], annot_kws={"size": 8},
        cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"],
        cbar = False,
        xticklabels=[str(xl.day) + " " + xl.month_name()[:3] for xl in xlabels],
        yticklabels = gantt_df["district"].unique(),
        cbar_kws = {
            "ticks":[0.5, 1, 2, 2.5], 
            "label": "Mobility", 
            "format": mlp.ticker.FuncFormatter(lambda x, pos: {0.5: "voluntary", 1: "cautionary", 2: "partial", 2.5: "restricted"}[x]), 
            "orientation": "horizontal", 
            "aspect": 50, 
            "drawedges": True,
            "fraction": 0.05,
            "pad": 0.10, 
            "shrink": 0.5
        }
    )
    # ax.set(xlabel = 'Days from '+start_date, ylabel = None)
    ax.set(xlabel = None, ylabel = None)
    plt.suptitle(title)
    plt.tight_layout()
    # plt.gcf().autofmt_xdate()
    plt.gcf().subplots_adjust(left=0.10, bottom=0.10)
# gantt_chart(gantt_data, "April 25, 2020", "Mobility Regime for Mumbai Wards (Daily Evaluation)")
# plt.show()
def run_policies(
        ward_cases:   Dict[str, pd.DataFrame], # timeseries for each ward 
        populations:  pd.Series,               # population for each ward
        wards:        Sequence[str],           # list of ward names 
        migrations:   np.matrix,               # O->D migration matrix, normalized
        gamma:        float,                   # 1/infectious period 
        Rmw:          Dict[str, float],        # mandatory regime R
        Rvw:          Dict[str, float],        # mandatory regime R
        total:        int   = 188*days,        # how long to run simulation
        eval_period:  int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling: float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:         int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 8 day lockdown 
    model_A = model(wards, populations, ward_cases, seed)
    simulate_lockdown(model_A, 8*days, total, Rmw, Rvw, lockdown, migrations)

    # 8 day + 4 week lockdown 
    model_B = model(wards, populations, ward_cases, seed)
    simulate_lockdown(model_B, 8*days + 4*weeks, total, Rmw, Rvw, lockdown, migrations)

    # 8 day lockdown + adaptive controls
    model_C = model(wards, populations, ward_cases, seed).set_parameters(RR0 = Rmw)
    simulate_adaptive_control(model_C, 8*days, total, lockdown, migrations, 
        {ward: beta_scaling * Rv * gamma for (ward, Rv) in Rvw.items()},
        {ward: beta_scaling * Rm * gamma for (ward, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C


if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    all_cases           = etl.load_case_data(data/"mumbai_wards_30Apr.csv")
    population_data     = etl.load_population_data(data/"ward_data_Mumbai_empt_slums.csv")
    (wards, migrations) = etl.load_migration_data(data/"Ward_rly_matrix_Mumbai.csv")
    lockdown = np.zeros(migrations.shape)

    ward_cases = {ward: all_cases[all_cases.ward == ward] for ward in wards}
    Rmw, Rvw = get_R(ward_cases, gamma)

    # model_A, model_B, model_C = run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, total = 188*days, eval_period = 1*days, seed = seed)

    # _, _, model_Cw = run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, total = 90*days, eval_period = 1*weeks, seed = seed)

    for i in range(1):
        model_A, model_B, model_C = run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, total = 188*days, eval_period = 2*weeks, seed = i)

    # plot_historical(all_cases, [model_A, model_B, model_C], 
    #     ["Release on 05 May", "Release on 02 Jun", "Adaptive Controls from 05 May"], 
    #     "Mumbai", None, "Daily Infections", "Ward-Level Adaptive Controls")
    # plt.show()

        gantt_chart(model_C.gantt, "April 25, 2020", "Release Schedule by Ward (Bi-Weekly Evaluation)")
        plt.show()

    # simulation_results = [ 
    #           run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, seed = seed)  
    #         for seed in range(100) 
    # ] 

    # I_100 = [tuple(m.aggregate("I") for m in ms) for ms in simulation_results]

    # Ia, Ib, Ic = zip(*I_100)

    # Ia_min = []
    # Ia_max = []
    # Ia_mdn = []

    # Ib_min = []
    # Ib_max = []
    # Ib_mdn = []

    # Ic_min = []
    # Ic_max = []
    # Ic_mdn = []
    # for t in range(189):
    #     Ia_sorted = sorted([ts[t] for ts in Ia if t < len(ts)])
    #     Ia_min.append(Ia_sorted[5])
    #     Ia_max.append(Ia_sorted[-5])
    #     Ia_mdn.append(Ia_sorted[50])

    #     Ib_sorted = sorted([ts[t] for ts in Ib if t < len(ts)])
    #     Ib_min.append(Ib_sorted[5])
    #     Ib_max.append(Ib_sorted[-5])
    #     Ib_mdn.append(Ib_sorted[50])
        
    #     Ic_sorted = sorted([ts[t] for ts in Ic if t < len(ts)])
    #     Ic_min.append(Ic_sorted[5])
    #     Ic_max.append(Ic_sorted[-5])
    #     Ic_mdn.append(Ic_sorted[50])

    # # fig = plt.figure()
    # th = all_cases[all_cases.ward == "ALL"].index

    # xp = [th.max() + pd.Timedelta(days = n) for n in range(189)]
    
    # plt.semilogy(th, all_cases["cases"][all_cases.ward == "ALL"], 'k-', label = "Empirical Case Data", linewidth = 3)

    # plt.semilogy(xp, Ia_mdn, label = "03 May Release", linewidth = 3)
    # plt.fill_between(xp, Ia_min, Ia_max, alpha = 0.5)
    
    # plt.semilogy(xp, Ib_mdn, label = "31 May Release", linewidth = 3)
    # plt.fill_between(xp, Ib_min, Ib_max, alpha = 0.5)

    # plt.semilogy(xp, Ic_mdn, label = "Adaptive Control", linewidth = 3)
    # plt.fill_between(xp, Ic_min, Ic_max, alpha = 0.5)
    
    # plt.xlabel("Date")
    # plt.ylabel("Number of Infections")
    
    # plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
    # plt.suptitle("Projected Infections over Time")
    # plt.title("Mumbai Wards")
    # plt.legend()
    # plt.show()