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
from etl import district_migration_matrices, get_time_series, load_data, v2
from adaptive.policy import simulate_adaptive_control_MHA #simulate_adaptive_control, simulate_adaptive_control_MHA

def simulate_adaptive_control(model: Model, initial_run: int, total_time: int, lockdown: np.matrix, migrations: np.matrix, beta_v: Dict[str, float], beta_m: Dict[str, float], evaluation_period = 2*weeks):
    n = len(model)
    # model.run(initial_run, lockdown)
    days_run = initial_run
    gantt = []
    last_category = dict()
    while days_run < total_time:
        Gs, Ys, Os, Rs = set(), set(), set(), set()
        categories = dict(enumerate([Gs, Ys, Os, Rs]))
        category_transitions = {}
        for (i, unit) in enumerate(model):
            latest_RR = unit.RR[-1]
            if latest_RR < 1: 
                Gs.add(i)
                beta_cat = 0
            else: 
                if days_run < evaluation_period + initial_run: # force first period to be lockdown
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
                    if i in categories[old_beta_cat]: categories[old_beta_cat].remove(i)
                    categories[beta_cat].add(i)
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
    gantt_df = gantt_df[gantt_df.day < 180]
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    start_timestamp = pd.to_datetime(start_date)
    levels = gantt_df.beta.unique()
    cmap = mlp.colors.ListedColormap(["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"])
    xlabels = [start_timestamp + pd.Timedelta(days = n) for n in gantt_df.day.unique()]
    ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        annot = gantt_pv["R"], annot_kws={"size": 8},
        cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"], #cmap,
        vmin = -1, 
        vmax = 4,
        cbar = False,
        xticklabels=[str(xl.day) + " " + xl.month_name()[:3] for xl in xlabels],
        yticklabels = gantt_df["district"].unique(),
    )
    # ax.set(xlabel = 'Days from '+start_date, ylabel = None)
    ax.set(xlabel = None, ylabel = None)
    plt.suptitle(title)
    plt.tight_layout()
    # plt.gcf().autofmt_xdate()
    plt.gcf().subplots_adjust(left=0.10, bottom=0.10, top = 0.94)

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
    total_time = 250 * days 
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
    grs = {state: run_regressions(timeseries, window = 5, infectious_period = 1/gamma) for (state, timeseries) in tss.items() if len(timeseries) > 5}
    
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

    districts, populations, migrations = migration_matrices[state]
    df_state = dfs[state]
    dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
    tsd = {district: get_time_series(cases) for (district, cases) in  dfd.items()}
    for (_, ts) in tsd.items():
        if 'Hospitalized' in ts:
            ts['Hospitalized'] *= prevalence
    grd = {district: run_regressions(timeseries, window = 5, infectious_period = 1/gamma) for (district, timeseries) in tsd.items() if len(timeseries) > 5}

    Rv = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else Rvs[state] for district in districts}
    Rm = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else Rms[state] for district in districts}

    # # fil in missing values 
    for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
        for key in mapping:
            if np.isnan(mapping[key]):
                mapping[key] = default

    # # policy scenarios: 
    lockdown = np.zeros(migrations.shape)

    ## adaptive release starting 03 may 
    beta_v = {district: R * gamma for (district, R) in Rv.items()}
    beta_m = {district: R * gamma for (district, R) in Rm.items()}
    np.random.seed(seed)
    adaptive = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
        .run(10, lockdown)\
        .set_parameters(RR0 = Rv)
    simulate_adaptive_control(adaptive, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=1*weeks)

    np.random.seed(seed)
    adaptive_MHA = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)\
        .run(10, lockdown)\
        .set_parameters(RR0 = Rv)
    simulate_adaptive_control_MHA(adaptive_MHA, 10*days, total_time, lockdown, migrations, beta_v, beta_m, evaluation_period=1*weeks)

    return (adaptive, adaptive_MHA)

if __name__ == "__main__":    
    sns.set(style="whitegrid", palette="bright", font="Fira Code")

    root = cwd()
    data = root/"data"

    states  = ["Kerala"]# ['Bihar', 'Andhra Pradesh',  'Tamil Nadu', 'Madhya Pradesh', 'Jammu and Kashmir'][3:]
    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    dfs = {state: dfn[dfn["detected state"] == state] for state in states}
    aggs = {state: [] for state in states}

    ts = dict()

    mods = []
    mhas = []

    for state in states: 
        for seed in tqdm(range(10)):
            (mod, mod_MHA) = run(112358 + seed, state)
            gantt_chart(mod_MHA.gantt, "April 23, 2020", f"Mobility Regime for {state} (MHA Starting Point)")
            plt.show()
            gantt_chart(mod.gantt, "April 23, 2020", f"Mobility Regime for {state} (Adaptive Control)")
            plt.show()

        

    # for (state, agg) in aggs.items():
    #     print(state)
    #     for (i, curve) in enumerate(agg): 
    #         print("  ", i, set(tup[2] for tup in curve.gantt))

    # for state in states[::-1]:
    #     for curve in aggs[state]:
    #         gantt_chart(curve.gantt, "April 23, 2020", f"Mobility Regime for {state} (MHA Starting Point)")
            

    # bh 6 
    # ap 0 
    # tn 2
    # kl 0 
    # mp 12 
    # jk 7

    

    # import json 
    # # json.dump(proj_a, open("proj_a.json", "w"))
    # # json.dump(proj_b, open("proj_b.json", "w"))
    # # json.dump(proj_c, open("proj_c.json", "w"))

    # proj_a = json.load((data/"proj_a.json").open())
    # proj_b = json.load((data/"proj_b.json").open())
    # proj_c = json.load((data/"proj_c.json").open())

    # states = ['Kerala', 'Rajasthan', 'Haryana',
    #    'Uttar Pradesh', 'Tamil Nadu','Jammu and Kashmir',
    #    'Karnataka', 'Maharashtra', 'Punjab', 'Andhra Pradesh',
       
    # states = [ 'Odisha',
    #        'Chhattisgarh', 'Gujarat', 'Himachal Pradesh', 'Madhya Pradesh',
    #    'Bihar','Meghalaya']
    # for state in states:
    #     gantt_chart(aggs[state][-6][-1].gantt, "April 23, 2020", f"Mobility Regime for {state} (Adaptive Lockdown)")
    #     plt.show()

        # print(state)
        # cases = dfs[state]
        # ts[state] = get_time_series(cases)

        # sns.set(style="whitegrid", palette="bright", font="Fira Code")

        # Ia_min = []
        # Ia_max = []
        # Ia_mdn = []

        # Ib_min = []
        # Ib_max = []
        # Ib_mdn = []

        # Ic_min = []
        # Ic_max = []
        # Ic_mdn = []

        # for t in range(240):
        #     Ia_sorted = sorted([ts[t] for ts in proj_a[state][0] if t < len(ts)])
        #     Ia_min.append(Ia_sorted[0])
        #     Ia_max.append(Ia_sorted[-1])
        #     Ia_mdn.append(Ia_sorted[len(Ia_sorted)//2])

        #     Ib_sorted = sorted([ts[t] for ts in proj_b[state][0] if t < len(ts)])
        #     Ib_min.append(Ib_sorted[0])
        #     Ib_max.append(Ib_sorted[-1])
        #     Ib_mdn.append(Ib_sorted[len(Ib_sorted)//2])
            
        #     Ic_sorted = sorted([ts[t] for ts in proj_c[state][0] if t < len(ts)])
        #     Ic_min.append(Ic_sorted[0])
        #     Ic_max.append(Ic_sorted[-1])
        #     Ic_mdn.append(Ic_sorted[len(Ic_sorted)//2])

        # th = ts[state].index

        # ts[state]["Hospitalized"].iloc[-1] = Ia_mdn[0]
        # plt.semilogy(th, ts[state]["Hospitalized"], 'k-', label = "Empirical Case Data", linewidth = 3)

        # xp = [th.max() + pd.Timedelta(days = n) for n in range(240)]
        # plt.semilogy(xp, Ia_mdn, label = "03 May Release", linewidth = 3)
        # plt.fill_between(xp, Ia_min, Ia_max, alpha = 0.3)
        
        # plt.semilogy(xp, Ib_mdn, label = "31 May Release", linewidth = 3)
        # plt.fill_between(xp, Ib_min, Ib_max, alpha = 0.3)

        # plt.semilogy(xp, Ic_mdn, label = "Adaptive Control", linewidth = 3)
        # plt.fill_between(xp, Ic_min, Ic_max, alpha = 0.3)
        
        # plt.xlabel("Date")
        # plt.ylabel("Number of Infections")
        
        # plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
        # plt.suptitle("Projected Infections over Time")
        # plt.title(state)
        # plt.legend()
        # plt.show()

