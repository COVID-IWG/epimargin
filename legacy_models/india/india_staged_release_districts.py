from itertools import product
from pathlib import Path
from typing import Dict, Sequence

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from india.growthratefit import *
from model import Model, ModelUnit

# sns.set_style("whitegrid")
# sns.set_palette("bright")
sns.set(style = "whitegrid", palette = "bright", font="Fira Code")
sns.despine()

def load_population_data(pop_path: Path) -> pd.DataFrame:
    return pd.read_csv(pop_path, names = ["name", "pop"])\
             .sort_values("name")

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M 

def load_district_migration_matrices(
    matrix_path: Path, 
    states: Sequence[str] = ["Maharashtra"]) -> Dict[str, np.matrix]:
    # states: Sequence[str] = ["Maharashtra", "Delhi", "Punjab", "Tamil Nadu", "Kerala"]) -> Dict[str, np.matrix]:
    mm = pd.read_csv(matrix_path)
    aggregations = dict()
    for col in  ['D_StateCensus2001', 'D_DistrictCensus2001', 'O_StateCensus2001', 'O_DistrictCensus2001']:
        mm[col] = mm[col].str.title().str.replace("&", "and")
    for state in  states:
        mm_state = mm[(mm.D_StateCensus2001 == state) & (mm.O_StateCensus2001 == state)]
        pivot    = mm_state.pivot(index = "D_DistrictCensus2001", columns = "O_DistrictCensus2001", values = "NSS_STMigrants").fillna(0)
        M  = np.matrix(pivot)
        Mn = M/M.sum(axis = 0)
        Mn[np.isnan(Mn)] = 0
        aggregations[state] = (
            pivot.index, 
            mm_state.groupby("O_DistrictCensus2001")["O_Population_2011"].agg(lambda x: list(x)[0]).values, 
            Mn
        )
    return aggregations 

if __name__ == "__main__":
    try: 
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path(".").resolve()
    
    data = root/"india"
    figs = root/"figures"
    
    seed = 25
    gamma = 1/5.0 
    prevalence = 1

    # run rolling regressions on historical case data 
    df_natl = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2)
    
    # can't do anything about this :(
    df_natl.dropna(subset = ["detected district"], inplace = True)

    df_natl["district"] = df_natl["detected district"].fillna("unknown") + "_" + df_natl["state code"]
    ts_natl = get_time_series(df_natl)
    gr_natl = run_regressions(ts_natl, window = 7, infectious_period = 5)
    
    states = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'][:2]
    # states = df_natl["detected state"].unique()

    # states = [_ for _ in df_natl["detected state"].dropna().unique() if _ not in ("Ladakh", "Uttarakhand")]
    dfs = {state: df_natl[df_natl["detected state"] == state] for state in states}
    tss = {state: get_time_series(state_data) for (state, state_data) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(state_data, window = 3, infectious_period = 5) for (state, state_data) in tss.items() if len(state_data) > 3}

    R_vol_natl  = np.mean(gr_natl.iloc[21:28].R)
    R_mand_natl = np.mean(gr_natl.iloc[28:-1].R)

    # get reproductive rate in realized policy regimes: pre- and post-national lockdown 
    R_vol_states  = {state: np.mean(grs[state].iloc[14:22].R) if state in grs else R_vol_natl  for state in states}
    R_mand_states = {state: np.mean(grs[state].iloc[22:-1].R) if state in grs else R_mand_natl for state in states}

    beta_vol_states  = {state: R * gamma  for  (state, R) in R_vol_states.items()}
    beta_mand_states = {state: R * gamma  for  (state, R) in R_mand_states.items()}

    migration_matrices = load_district_migration_matrices(data/"Migration Matrix - District.csv", states = states)
    for state in states:
        districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
        tsd = {district: get_time_series(district_data) for (district, district_data) in  dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(district_data, window = 3, infectious_period = 5) for (district, district_data) in tsd.items() if len(district_data) > 3}

        # print({k: len(v) for (k, v) in tsd.items()})

        R_vol  = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else R_vol_states[state]  for district in districts}
        R_mand = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else R_mand_states[state] for district in districts}

        for mapping, default in ((R_vol, R_vol_states[state]), (R_mand, R_mand_states[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default

        beta_vol  = { district: R * gamma for (district, R) in R_vol.items()  }
        beta_mand = { district: R * gamma for (district, R) in R_mand.items() }

        def get_units(): 
            return [ModelUnit(
                name = district, 
                population = populations[i], 
                I0  = tsd[district].iloc[-1]['Hospitalized'] if not tsd[district].empty and 'Hospitalized' in tsd[district].iloc[-1] else 0,
                R0  = tsd[district].iloc[-1]['Recovered']    if not tsd[district].empty and 'Recovered'    in tsd[district].iloc[-1] else 0,
                D0  = tsd[district].iloc[-1]['Deceased']     if not tsd[district].empty and 'Deceased'     in tsd[district].iloc[-1] else 0,
                RR0 = R_mand[district]) 
            for (i, district) in enumerate(districts)]

        lockdown = np.zeros(migrations.shape)

        # policy A: end lockdown on May 3 
        np.random.seed(seed)
        model_A_1 = Model(num_days = 10, units = get_units(), migrations = lockdown).run()
        for unit in model_A_1: 
            unit.RR0 = R_vol[unit.name]
        model_A_2 = Model(num_days = 180, units = model_A_1.units, migrations = migrations).run()


        # policy B: end lockdown May 31
        np.random.seed(seed)
        model_B_1 = Model(num_days = 38, units = get_units(), migrations = lockdown).run()
        for unit in model_B_1: 
            unit.RR0 = R_vol[unit.name]
        model_B_2 = Model(num_days = 152, units = model_B_1.units, migrations = migrations).run()

        # policy C: 3 phased release 
        np.random.seed(seed)
        units_C = get_units()
        n = len(units_C)
        
        model_C = Model(num_days = 10, units = units_C, migrations = lockdown).run()
        phased_migration = lockdown.copy()
        days_run = 10
        gantt = []
        last_category = {}
        while days_run < 190:
            Gs, Ys, Os, Rs = set(), set(), set(), set()
            category_transitions = {}
            for (i, unit) in enumerate(model_C):
                latest_RR = unit.RR[-1]
                if latest_RR < 1: 
                    Gs.add(i)
                    beta_cat = 0
                else: 
                    if days_run < 14: # force first period to be lockdown
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
                unit =  model_C[unit_name]
                new_beta = beta_vol[unit.name] - (beta_cat * (beta_vol[unit.name] - beta_mand[unit.name])/3.0)                
                unit.beta[-1] = new_beta
                unit.RR0 = new_beta * unit.gamma

            phased_migration = migrations.copy()
            for (i, j) in product(range(n), range(n)):
                if i not in Gs or j not in Gs:
                    phased_migration[i, j] = 0
            model_C = Model(num_days = 14, units = units_C, migrations = phased_migration).run()
            days_run += 14

        # plot gantt chart
        gantt_df = pd.DataFrame(gantt, columns = ["district", "day", "beta", "R"])
        gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])

        path = figs/(f"india/")
        if not path.exists():
            path.mkdir()

        # infection curves
        # fig = plt.figure()
        # fig.set_size_inches(8, 6)
        # plt.semilogy(model_A_2.aggregate("I"), label = "03 May release", figure = fig) 
        # plt.semilogy(model_B_2.aggregate("I"), label = "31 May release", figure = fig) 
        # plt.semilogy(model_C.aggregate("I"), label = "adaptive release", figure = fig) 
        # plt.suptitle(state) 
        # if prevalence > 1:
        #     plt.title(f"True Prevalence = {prevalence} × Confirmed Cases")
        # plt.xlabel("Days Since April 23") 
        # plt.ylabel("Number of Infections") 
        # plt.legend() 
        # plt.tight_layout()
        # filename = "scenarios_" + state.lower().replace(" ", "_")  + ".png"
        # plt.savefig(path/filename, bbox_inches="tight", dpi = 600)
        # # plt.show() 
        # plt.close()

        # # R vs I
        fig = plt.figure()
        # fig.set_size_inches(8, 6)
        rep_unit = max(model_A_2.units, key = lambda u: len(tsd[u.name])).name
        # plt.plot(model_A_2[rep_unit].I[1:], model_A_2[rep_unit].b[1:])
        plt.plot(list(map(lambda i: i[1] - i[0], zip(model_A_2[rep_unit].I[1:], model_A_2[rep_unit].I[:-1]))), model_A_2[rep_unit].b[1:], label = "03 May release", figure = fig) 
        # plt.plot(list(map(lambda i: i[1] - i[0], zip(model_B_2[rep_unit].I[1:], model_B_2[rep_unit].I[:-1]))), model_B_2[rep_unit].beta[1:], label = "31 May release", figure = fig) 
        # plt.plot(list(map(lambda i: i[1] - i[0], zip(model_C[rep_unit].I[1:], model_C[rep_unit].I[:-1]))), model_C[rep_unit].beta[1:], label = "adaptive release", figure = fig) 
        # plt.xlabel(r"$\frac{dI}{dt}$") 
        # plt.ylabel(r"$\beta$") 
        plt.title(f"({rep_unit}, {state})")
        plt.legend()
        plt.tight_layout()
        filename = "beta_dIdt_" + state.lower().replace(" ", "_")  + ".png"
        # plt.savefig(path/filename, bbox_inches="tight", dpi = 600)
        plt.show()
        plt.close()

        # ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        #     annot = gantt_pv["R"], annot_kws={"size": 8},
        #     # cmap = ["#2ecc71", "#fada4c","#eda81f", "#cc2e2e"], #square = True, 
        #     cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"],
        #     cbar = state != "Uttar Pradesh",
        #     # yticklabels = districts,
        #     cbar_kws = {
        #         "ticks":[0.5, 1, 2, 2.5], 
        #         "label": "Mobility", 
        #         "format": mlp.ticker.FuncFormatter(lambda x, pos: {0.5: "voluntary", 1: "cautionary", 2: "partial", 2.5: "restricted"}[x]), 
        #         "orientation": "horizontal", 
        #         "aspect": 50, 
        #         "drawedges": True,
        #         "fraction": 0.05,
        #         "pad": 0.10, 
        #         "shrink": 0.5
        #         })
        # ax.set(xlabel = "Days from 23 Apr", ylabel = None)
        # plt.suptitle(f"Rolling Release Scenarios for {state} (Annotated by Reproductive Number)")
        # plt.tight_layout()
        # plt.gcf().subplots_adjust(left=0.10, bottom=0.10)
        # filename = "gantt_" + state.lower().replace(" ", "_") + ".png"
        # plt.savefig(path/filename, dpi=600, bbox_inches="tight")
        # # plt.show()
        # plt.close()

        