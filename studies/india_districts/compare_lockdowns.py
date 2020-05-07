from itertools import product
from pathlib import Path
from typing import Dict, Sequence, Optional

import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from adaptive.estimators import rollingOLS as run_regressions
from adaptive.model import Model, ModelUnit
# from adaptive.plotting import plot_curve, gantt_chart
from adaptive.utils import *
from adaptive.policy import * 
from etl import district_migration_matrices, get_time_series, load_data, v2

def plot_historical(all_cases: pd.DataFrame, models: Sequence[Model], labels: Sequence[str], title, xlabel, ylabel, subtitle = None, curve: str = "I", filename = None):
    fig = plt.figure()
    th = all_cases.index
    plt.semilogy(th, all_cases["Hospitalized"], 'k-', label = "Empirical Case Data", figure = fig, linewidth = 3)
    
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
    # fig.autofmt_xdate()
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

def get_model(districts, populations, timeseries, seed = 0):
    units = [ModelUnit(
        name       = district, 
        population = populations[i],
        I0  = timeseries[district].iloc[-1]['Hospitalized'] if not timeseries[district].empty and 'Hospitalized' in timeseries[district].iloc[-1] else 0,
        R0  = timeseries[district].iloc[-1]['Recovered']    if not timeseries[district].empty and 'Recovered'    in timeseries[district].iloc[-1] else 0,
        D0  = timeseries[district].iloc[-1]['Deceased']     if not timeseries[district].empty and 'Deceased'     in timeseries[district].iloc[-1] else 0,
    ) for (i, district) in enumerate(districts)]
    return Model(units, random_seed = seed)

# lockdown policy of different lengths
def simulate_lockdown(model: Model, lockdown_period: int, total_time: int, RR0_mandatory: Dict[str, float], RR0_voluntary: Dict[str, float], lockdown: np.matrix, migrations: np.matrix) -> Model:
    return model.set_parameters(RR0 = RR0_mandatory)\
        .run(lockdown_period,  migrations = lockdown)\
        .set_parameters(RR0 = RR0_voluntary)\
        .run(total_time - lockdown_period, migrations = migrations)

# policy C: adaptive release
def simulate_adaptive_control(model: Model, initial_run: int, total_time: int, lockdown: np.matrix, migrations: np.matrix, beta_v: Dict[str, float], beta_m: Dict[str, float], evaluation_period = 1*weeks):
    n = len(model)
    # model.run(initial_run, lockdown)
    days_run = initial_run
    gantt = []
    last_category = dict()
    while days_run < total_time:
        Gs, Ys, Os, Rs = set(), set(), set(), set()
        category_transitions = {}
        for (i, unit) in enumerate(model):
            latest_RR = unit.RR[-1]
            if latest_RR < 1: 
                Gs.add(i)
                beta_cat = 0
            else: 
                if days_run < evaluation_period + days_run: # force first period to be lockdown
                    Rs.add(i)
                    beta_cat = 3
                else: 
                    if latest_RR < 1.5: 
                        Ys.add(i)
                        beta_cat = 1
                    elif latest_RR < 2: 
                        Os.add(i)
                        beta_cat = 2
                    else:
                        Rs.add(i)
                        beta_cat = 3
            if unit.name not in last_category:
                last_category[unit.name] = beta_cat
            else: 
                old_beta_cat = last_category[unit.name]
                if old_beta_cat != beta_cat:
                    if beta_cat < old_beta_cat and beta_cat != (old_beta_cat - 1): # force gradual release
                        beta_cat = old_beta_cat - 1
                    category_transitions[unit.name] = beta_cat
                    last_category[unit.name] = beta_cat 
            gantt.append([unit.name, days_run, beta_cat, max(0, latest_RR)])

        for (unit_name, beta_cat) in category_transitions.items(): 
            unit =  model[unit_name]
            new_beta = beta_v[unit.name] - (beta_cat * (beta_v[unit.name] - beta_m[unit.name])/3.0)                
            unit.beta[-1] = new_beta
            unit.RR0 = new_beta * unit.gamma

        phased_migration = migrations.copy()
        for (i, j) in product(range(n), range(n)):
            if i not in Gs or j not in Gs:
                phased_migration[i, j] = 0
        model.run(evaluation_period, phased_migration)
        days_run += evaluation_period

    model.gantt = gantt 
    return model 

def run(seed, state):
    # set up folders
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    if not figs.exists():
        figs.mkdir()

    # simulation parameters 
    # seed       = 0
    # total_time = 190 * days 
    total_time = 190 * days 
    # states     = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'][2:3]
    states = [state]
    
    # model details 
    gamma      = 0.2
    prevalence = 1

    # run rolling regressions on historical national case data 
    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    tsn = get_time_series(dfn)
    grn = run_regressions(tsn, window = 7, infectious_period = 1/gamma)

    # disaggregate down to states
    dfs = {state: dfn[dfn["detected state"] == state] for state in states}
    tss = {state: get_time_series(cases) for (state, cases) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (state, timeseries) in tss.items() if len(timeseries) > 3}
    
    # voluntary and mandatory reproductive numbers
    Rvn = np.mean(grn["2020-03-24":"2020-03-31"].R)
    Rmn = np.mean(grn["2020-04-01":].R)
    Rvs = {s: np.mean(grs[s]["2020-03-24":"2020-03-31"].R) if s in grs else Rvn for s in states}
    Rms = {s: np.mean(grs[s]["2020-04-01":].R)             if s in grs else Rmn for s in states}

    # voluntary and mandatory distancing rates 
    Bvs = {s: R * gamma for (s, R) in Rvs.items()}
    Bms = {s: R * gamma for (s, R) in Rms.items()}

    migration_matrices = district_migration_matrices(data/"Migration Matrix - District.csv", states = states)

    out = dict()

    for state in states: 
        districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
        tsd = {district: get_time_series(cases) for (district, cases) in  dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (district, timeseries) in tsd.items() if len(timeseries) > 3}
    
        Rv = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else Rvs[state] for district in districts}
        Rm = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else Rms[state] for district in districts}

        # # fil in missing values 
        for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default

        # # policy scenarios: 
        lockdown = np.zeros(migrations.shape)

        # ## release lockdown on 03 May 
        release_03_may = get_model(districts, populations, tsd, seed)
        simulate_lockdown(release_03_may, 
            lockdown_period = 10*days, 
            total_time      = total_time, 
            RR0_mandatory   = Rm,              RR0_voluntary = Rv, 
            lockdown        = lockdown.copy(), migrations    = migrations)

        ## release lockdown on 31 May 
        release_31_may = get_model(districts, populations, tsd, seed)
        simulate_lockdown(release_31_may, 
            lockdown_period = 10*days + 4*weeks, 
            total_time      = total_time, 
            RR0_mandatory   = Rm,              RR0_voluntary = Rv, 
            lockdown        = lockdown.copy(), migrations    = migrations)

        ## adaptive release starting 03 may 
        beta_v = {district: R * gamma for (district, R) in Rv.items()}
        beta_m = {district: R * gamma for (district, R) in Rm.items()}
        adaptive = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
            .run(10, lockdown)\
            .set_parameters(RR0 = Rv)
        simulate_adaptive_control(adaptive, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=2*weeks)
        # plot_historical(tss[state],
        #     [release_03_may, release_31_may, adaptive], 
        #     ["03 May Release", "31 May Release", "Adaptive Release"], 
        #     title = state, xlabel = "Date", ylabel = "Number of Infections", subtitle = None
        # )

        # plt.show()

        # gantt_chart(adaptive.gantt, "Days Since April 23", "Release Strategy")
        
        # plt.show()
        out[state] = ((release_03_may, release_31_may, adaptive))
    return out 


def run_ACs(seed, state):
    # set up folders
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    if not figs.exists():
        figs.mkdir()

    # simulation parameters 
    # seed       = 0
    # total_time = 190 * days 
    total_time = 190 * days 
    # states     = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'][2:3]
    states = [state]
    
    # model details 
    gamma      = 0.2
    prevalence = 1

    # run rolling regressions on historical national case data 
    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    tsn = get_time_series(dfn)
    grn = run_regressions(tsn, window = 7, infectious_period = 1/gamma)

    # disaggregate down to states
    dfs = {state: dfn[dfn["detected state"] == state] for state in states}
    tss = {state: get_time_series(cases) for (state, cases) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (state, timeseries) in tss.items() if len(timeseries) > 3}
    
    # voluntary and mandatory reproductive numbers
    Rvn = np.mean(grn["2020-03-24":"2020-03-31"].R)
    Rmn = np.mean(grn["2020-04-01":].R)
    Rvs = {s: np.mean(grs[s]["2020-03-24":"2020-03-31"].R) if s in grs else Rvn for s in states}
    Rms = {s: np.mean(grs[s]["2020-04-01":].R)             if s in grs else Rmn for s in states}

    # voluntary and mandatory distancing rates 
    Bvs = {s: R * gamma for (s, R) in Rvs.items()}
    Bms = {s: R * gamma for (s, R) in Rms.items()}

    migration_matrices = district_migration_matrices(data/"Migration Matrix - District.csv", states = states)

    out = dict()

    for state in states: 
        districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
        tsd = {district: get_time_series(cases) for (district, cases) in  dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (district, timeseries) in tsd.items() if len(timeseries) > 3}
    
        Rv = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else Rvs[state] for district in districts}
        Rm = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else Rms[state] for district in districts}

        # # fil in missing values 
        for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default

        # # policy scenarios: 
        lockdown = np.zeros(migrations.shape)
        beta_v = {district: R * gamma for (district, R) in Rv.items()}
        beta_m = {district: R * gamma for (district, R) in Rm.items()}
        
        adaptive1 = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
            .run(10, lockdown)\
            .set_parameters(RR0 = Rv)
        simulate_adaptive_control(adaptive1, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=2*weeks)
        
        adaptive2 = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
            .run(10, lockdown)\
            .set_parameters(RR0 = Rv)
        simulate_adaptive_control(adaptive2, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=2*weeks)
        
        adaptive3 = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
            .run(10, lockdown)\
            .set_parameters(RR0 = Rv)
        simulate_adaptive_control(adaptive3, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=2*weeks)
        
        adaptive4 = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
            .run(10, lockdown)\
            .set_parameters(RR0 = Rv)
        simulate_adaptive_control_MHA(adaptive4, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=2*weeks)
        # plot_historical(tss[state],
        #     [release_03_may, release_31_may, adaptive], 
        #     ["03 May Release", "31 May Release", "Adaptive Release"], 
        #     title = state, xlabel = "Date", ylabel = "Number of Infections", subtitle = None
        # )

        # plt.show()

        # gantt_chart(adaptive.gantt, "Days Since April 23", "Release Strategy")
        
        # plt.show()
        out[state] = (adaptive1, adaptive2, adaptive3, adaptive4)
    return out 

if __name__ == "__main__":

    # for seed in (108, 2000, 25, 33): 
    #     ma, mb, mc = run(seed, "Bihar")["Bihar"]
    #     gantt_chart(mc.gantt, "April 23, 2020", "Mobility Regime for Bihar Districts (Bi-Weekly Evaluation)")
    #     plt.subplots_adjust(0.08, 0.08, 0.97, 0.94)
    #     plt.show()
    
    sns.set(style="whitegrid", font="Fira Code")
    root = cwd()
    data = root/"data"
    # states     = ["Tamil Nadu", 'Bihar', 'Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'West Bengal', 'Kerala', 'Gujarat']
    states     = ['Bihar', 'Punjab', 'Kerala']
    aggs = {state: [] for state in states}

    ld_a = {state: [] for state in states}
    ld_b = {state: [] for state in states}
    ld_c = {state: [] for state in states}
    ld_d = {state: [] for state in states}
    
    ts = dict()

    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    dfs = {state: dfn[dfn["detected state"] == state] for state in states}

    for state in states: 
        for seed in tqdm(range(10)):
            run_out = run_ACs(seed, state)
            for state in run_out.keys():
                aggs[state].append(run_out[state])
        ma, mb, mc, md = list(zip(*aggs[state]))
        ld_a[state].append([m.aggregate("I") for m in ma])
        ld_b[state].append([m.aggregate("I") for m in mb])
        ld_c[state].append([m.aggregate("I") for m in mc])
        ld_d[state].append([m.aggregate("I") for m in md])
        # Ic = [m.aggregate("I") for m in mc]

    # import json 
    # json.dump(ld_a, open("ld_a.json", "w"))
    # json.dump(ld_b, open("ld_b.json", "w"))
    # json.dump(ld_c, open("ld_c.json", "w"))

    for state in states:
        cases = dfs[state]
        ts[state] = get_time_series(cases)

        Ia_min = []
        Ia_max = []
        Ia_mdn = []

        Ib_min = []
        Ib_max = []
        Ib_mdn = []

        Ic_min = []
        Ic_max = []
        Ic_mdn = []

        Id_min = []
        Id_max = []
        Id_mdn = []

        for t in range(120):
            Ia_sorted = sorted([ts[t] for ts in ld_a[state][0] if t < len(ts)])
            Ia_min.append(Ia_sorted[0])
            Ia_max.append(Ia_sorted[-1])
            Ia_mdn.append(Ia_sorted[len(Ia_sorted)//2])

            Ib_sorted = sorted([ts[t] for ts in ld_b[state][0] if t < len(ts)])
            Ib_min.append(Ib_sorted[0])
            Ib_max.append(Ib_sorted[-1])
            Ib_mdn.append(Ib_sorted[len(Ib_sorted)//2])
            
            Ic_sorted = sorted([ts[t] for ts in ld_c[state][0] if t < len(ts)])
            Ic_min.append(Ic_sorted[0])
            Ic_max.append(Ic_sorted[-1])
            Ic_mdn.append(Ic_sorted[len(Ic_sorted)//2])
            
            Id_sorted = sorted([ts[t] for ts in ld_d[state][0] if t < len(ts)])
            Id_min.append(Id_sorted[0])
            Id_max.append(Id_sorted[-1])
            Id_mdn.append(Id_sorted[len(Id_sorted)//2])

        th = ts[state].index

        ts[state]["Hospitalized"].iloc[-1] = Ia_mdn[0]
        plt.semilogy(th, ts[state]["Hospitalized"], 'k-', label = "Empirical Case Data", linewidth = 3)

        xp = [th.max() + pd.Timedelta(days = n) for n in range(120)]
        plt.semilogy(xp, Ia_mdn, label = "AC (cutoffs = 1, 1.5, 2)", linewidth = 3, color = "#01C23B")
        plt.fill_between(xp, Ia_min, Ia_max, alpha = 0.1, color = "#01C23B")
        
        plt.semilogy(xp, Ib_mdn, label = "AC (cutoffs = 1, 1.2, 1.5)", linewidth = 3, color = "#9b59b6")
        plt.fill_between(xp, Ib_min, Ib_max, alpha = 0.1, color = "#9b59b6")

        plt.semilogy(xp, Ic_mdn, label = "AC (cutoffs = 1, 1.8, 2.5)", linewidth = 3, color = "#e74c3c")
        plt.fill_between(xp, Ic_min, Ic_max, alpha = 0.1, color = "#e74c3c")

        plt.semilogy(xp, Id_mdn, label = "AC (MHA starting point)", linewidth = 3, color = "#881161")
        plt.fill_between(xp, Id_min, Id_max, alpha = 0.1, color = "#881161")
        
        plt.xlabel("Date")
        plt.ylabel("Number of Infections")
        
        plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
        plt.suptitle("Projected Infections over Time")
        plt.title(state)
        plt.legend()
        plt.show()

        # gantt_chart(aggs[state][-1][0].gantt, "April 23, 2020", f"Mobility Regime for {state} Districts (Bi-Weekly Evaluation, Adaptive Control)")
        # plt.show()
        # gantt_chart(aggs[state][-1][-1].gantt, "April 23, 2020", f"Mobility Regime for {state} Districts (Bi-Weekly Evaluation, MHA Starting Point)")
        # plt.show()
