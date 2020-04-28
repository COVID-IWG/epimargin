from itertools import product
from pathlib import Path

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from india.growthratefit import *
from model import Model, ModelUnit

sns.set(style = "whitegrid", font="Fira Code")
# sns.set_style("white")
sns.set_palette("bright")
sns.despine()

def create_empirical_migration_matrix(
    census_path:     Path, 
    migrations_path: Path, 
    output_path    : Path) -> pd.DataFrame:
    # load in census metadata 
    census = pd.read_csv(census_path, use_cols = "District")
    splits = census["District"].str.split(",", expand=True)
    # load in observed migrations 

def load_population_data(pop_path: Path) -> pd.DataFrame:
    return pd.read_csv(pop_path, names = ["name", "pop"])\
             .sort_values("name")

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M 

if __name__ == "__main__":
    try: 
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path(".").resolve()
    
    data = root/"india"
    figs = root/"figures"

    # run rolling regressions on historical case data 
    df_natl = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2)
    
    # can't do anything about this :(
    df_natl.dropna(subset = ["detected district"], inplace = True)

    # disambiguate districts with the same name in different states 
    df_natl["district"] = df_natl["detected district"].fillna("unknown") + "_" + df_natl["state code"]
    ts_natl = get_time_series(df_natl)
    gr_natl = run_regressions(ts_natl, window = 7, infectious_period = 5)


    # migrations
    # migrations = load_empirical_migration_matrix(data/"census_2001_kaggle.csv", data/"6_Migration Matrix - District.csv")

    # states = (MH, NCT, KL, UP) = ["Maharashtra", "Delhi", "Kerala", "Uttar Pradesh"]
    states = [_ for _ in df_natl["detected state"].dropna().unique() if _ not in ("Ladakh", "Uttarakhand")]
    dfs = {state: df_natl[df_natl["detected state"] == state] for state in states}
    tss = {state: get_time_series(state_data) for (state, state_data) in dfs.items()}
    grs = {state: run_regressions(state_data, window = 7, infectious_period = 5) for (state, state_data) in tss.items() if len(state_data) > 7}

    # states = grs.keys()

    R_vol_natl  = np.mean(gr_natl.iloc[21:28].R)
    R_mand_natl = np.mean(gr_natl.iloc[28:-1].R)

    # get reproductive rate in realized policy regimes: pre- and post-national lockdown 
    R_vol  = {state: np.mean(grs[state].iloc[14:22].R) if state in grs else R_vol_natl  for state in states}
    R_mand = {state: np.mean(grs[state].iloc[22:-1].R) if state in grs else R_mand_natl for state in states}

    pop_data   = load_population_data(data/"india_pop.csv") 
    migrations = load_migration_matrix(data/"india_migration_matrix.csv", np.array(pop_data["pop"]))
    lockdown   = np.zeros(migrations.shape)
    
    pop_data   = {name: pop for (name, pop) in pop_data.itertuples(index = False)}
    pop_data["Odisha"] = pop_data.pop("Orissa")
    pop_data["Puducherry"] = pop_data.pop("Pondicherry")

    # set nans to natl 
    for mapping, default in ((R_vol, R_vol_natl), (R_mand, R_mand_natl)):
        for key in mapping:
            if np.isnan(mapping[key]):
                mapping[key] = default
        for key in [key for key in pop_data.keys() if key not in mapping]:
            mapping[key] = default

    gamma = 1/5.0 
    beta_vol  = {state: R * gamma  for  (state, R) in R_vol.items()}
    beta_mand = {state: R * gamma  for  (state, R) in R_mand.items()}

    def get_model_unit(state, pop, RR0):
        try: 
            return ModelUnit(
                name = state, population = pop, 
                I0  = tss[state].iloc[-1]['Hospitalized'] if 'Hospitalized' in tss[state].iloc[-1] else 0,
                R0  = tss[state].iloc[-1]['Recovered']    if 'Recovered'    in tss[state].iloc[-1] else 0,
                D0  = tss[state].iloc[-1]['Deceased']     if 'Deceased'     in tss[state].iloc[-1] else 0,
                RR0 = RR0
            )
        except KeyError:
            return ModelUnit(state, pop, 0, 0, 0, RR0)

    seed = 11235813

    # policy A: end lockdown on May 3 
    np.random.seed(seed)
    units_A = [get_model_unit(state, pop_data[state], R_mand[state]) for state in pop_data.keys()]
    model_A_1 = Model(num_days = 10, units = units_A, migrations = lockdown).run()
    for unit in units_A: 
        unit.RR0 = R_vol[unit.name]
    model_A_2 = Model(num_days = 120, units = units_A, migrations = migrations).run()

    # policy B: end lockdown May 31
    np.random.seed(seed)
    units_B = [get_model_unit(state, pop_data[state], R_mand[state]) for state in pop_data.keys()]
    model_B_1 = Model(num_days = 38, units = units_B, migrations = lockdown).run()
    for unit in units_B: 
        unit.RR0 = R_vol[unit.name]
    model_B_2 = Model(num_days = 92, units = units_B, migrations = migrations).run()

    # policy C: 3 phased release 
    np.random.seed(seed)
    units_C = [get_model_unit(state, pop_data[state], R_mand.get(state, R_mand_natl)) for state in pop_data.keys()]
    n = len(units_C)
    # up to may 3: full lockdown: 
    model_C = Model(num_days = 10, units = units_C, migrations = lockdown).run()
    phased_migration = lockdown.copy()
    days_run = 10
    gantt = []
    last_category = {}
    while days_run < 130:
        Gs, Ys, Rs = set(), set(), set()
        category_transitions = {}
        for (i, unit) in enumerate(model_C):
            latest_RR = unit.RR[-1]
            if latest_RR < 1: 
                Gs.add(i)
                beta_cat = 1
            elif latest_RR < 2: 
                Ys.add(i)
                beta_cat = 2
            else: 
                Rs.add(i)
                beta_cat = 3
            gantt.append([unit.name, days_run, beta_cat, max(0, latest_RR)])
            if unit.name not in last_category:
                last_category[unit.name] = beta_cat
            else: 
                if last_category[unit.name] != beta_cat:
                    category_transitions[unit.name] = beta_cat
                    last_category[unit.name] = beta_cat 

        for (unit_name, beta_cat) in category_transitions.items(): 
            unit =  model_C[unit_name]
            new_beta = beta_mand[unit.name] + (3 - beta_cat) * (beta_vol[unit.name] - beta_mand[unit.name])/3.0
            unit.beta[-1] = new_beta
            unit.RR0 = new_beta * unit.gamma

        phased_migration = migrations.copy()
        for (i, j) in product(range(n), range(n)):
            if i in Rs or j in Rs:
                phased_migration[i, j] = 0
        model_C = Model(num_days = 14, units = units_C, migrations = phased_migration).run()
        days_run += 14

    # seedpath = figs/("india/seed" + str(seed))
    # if not seedpath.exists():
    #     seedpath.mkdir()
    # for (A, B, C) in zip(model_A_2, model_B_2, model_C):
    #     plt.figure()
    #     plt.semilogy(A.I, label = "Policy A") 
    #     plt.semilogy(B.I, label = "Policy B") 
    #     plt.semilogy(C.I, label = "Policy C") 
    #     plt.title(A.name) 
    #     plt.xlabel("Days Since April 23") 
    #     plt.ylabel("Number of Infections") 
    #     plt.legend() 
    #     plt.tight_layout()
    #     filename = "scenarios_" + A.name.lower().replace(" ", "_")  + ".png"
    #     plt.savefig(seedpath/filename, bbox_inches="tight", dpi = 600)
    #     # plt.show() 
    #     plt.close()

    # for state in ("Gujarat", "Tamil Nadu", "Maharashtra", "Uttar Pradesh", "Punjab", "West Bengal"):
    #     A, B, C = [model[state] for model in (model_A_2, model_B_2, model_C)]
    #     plt.figure()
    #     plt.semilogy(A.I, color = "#7F7F7F", linewidth = 2)
    #     plt.semilogy(B.I, color = "#0D0D0D", linewidth = 2)
    #     plt.semilogy(C.I, color = "#A5B4B2", linewidth = 4)
    #     plt.semilogy(C.I, color = "#6A7F7B", linewidth = 3)
    #     plt.ylim(1, 10**8)
    #     plt.xlim(0, 120)
    #     plt.tight_layout()
    #     # plt.axes().set_aspect('equal')
    #     plt.savefig(f"/Users/satej/Documents/workspace/mnp/capp_pres/preso_{state}.png", dpi=600, bbox_inches="tight", transparent = True)
    #     plt.close()
    #     # plt.title(state)
    #     # plt.show()

    # for (A, B, C) in zip(model_A_2, model_B_2, model_C):
    #     pop = pop_data[A.name]
    #     plt.plot([i/pop for i in A.I], label = "Policy A")
    #     plt.plot([i/pop for i in B.I], label = "Policy B")
    #     plt.plot([i/pop for i in C.I], label = "Policy C")
    #     plt.title(A.name)
    #     plt.xlabel("Days Since April 23")
    #     plt.ylabel("Infection Rate")
    #     plt.legend()
    #     plt.show()
