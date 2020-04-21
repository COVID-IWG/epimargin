from pathlib import Path

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from india.growthratefit import *
from model import Model, ModelUnit

mlp.rcParams['font.sans-serif'] = "PT Sans"
mlp.rcParams['font.family'] = "sans-serif"
font = {'family': 'sans-serif', 'sans-serif': ["PT Sans"]}
mlp.rc('font', **font)
sns.set_style("whitegrid")
sns.set_palette("bright")
sns.despine()

def load_population_data(pop_path: Path) -> pd.DataFrame:
    return pd.read_csv(india_data/"india_pop.csv", names = ["name", "pop"])\
             .sort_values("name")

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M 

def get_model_unit(name: str, pop: float, RR0: float = 1.9, I0: int = 0, R0: int = 0, D0: int = 0) -> ModelUnit:
    unit = ModelUnit(name, pop, RR0 = RR0)
    if any(v != 0 for v in (I0, R0, D0)):
        unit.I[0] = I0
        unit.R[0] = R0
        unit.D[0] = D0 
        unit.P[0] -= (I0 + R0 + D0)
    return unit 

if __name__ == "__main__":
    root = Path(__file__).parent
    india_data = root/"india"

    # run rolling regressions on historical case data 
    df_natl = load_data(india_data/"india_case_data_resave.csv", reduced = True)
    ts_natl = get_time_series(df_natl)
    gr_natl = run_regressions(ts_natl)

    RR0 = gr_natl.iloc[-1].R # it's about 1.9

    # load data for epi model
    pop_data   = load_population_data(india_data/"india_pop.csv")
    migrations = load_migration_matrix(india_data/"india_migration_matrix.csv", np.array(pop_data["pop"]))

    # build units 
    units    = []
    state_ts = dict()
    for (name, pop) in pop_data.itertuples(index = False):
        ts = get_time_series(df_natl[df_natl["detected state"] == name])
        state_ts[name] = ts 
        if len(ts) > 0:
            latest_counts =  ts.iloc[-1]
            I0, R0, D0 = (
                latest_counts['Hospitalized'] if 'Hospitalized' in latest_counts else 0, 
                latest_counts['Recovered']    if 'Recovered'    in latest_counts else 0, 
                latest_counts['Deceased']     if 'Deceased'     in latest_counts else 0
            )
        else:
            I0, R0, D0 = 0, 0, 0
        unit = get_model_unit(name, pop, RR0, I0, R0, D0)
        units.append(unit)
    
    model = Model(200, units, migrations).run()
    mh = next(_ for _ in model.units if _.name == "Maharashtra")

    # plot data for MH 
    # reproductive rate plot 
    ts_mh = state_ts["Maharashtra"]
    gr_mh = run_regressions(ts_mh)
    
    RR = np.concatenate([np.array(gr_mh.R), np.array(mh.RR)])
    fig = plt.figure()
    ax  = plt.gca()
    plt.suptitle("Maharashtra Migration Example: Reproductive Rate")
    plt.title("Lockdown Lifted: March 31st")
    rr = plt.plot(RR)
    plt.plot([len(ts_mh), len(ts_mh)], [-3, 9], 'k--', alpha = 0.5, linewidth = 1)
    ax.set(xlabel = "# days since March 10", ylabel = "Reproductive Rate $R(t)$")
    ax.label_outer()

    # SIRD plots 
    S = np.concatenate([np.array([mh.pop0] * len(ts_mh)), np.array(mh.S)])
    I = np.concatenate([np.array(ts_mh["Hospitalized"]),  np.array(mh.I)])
    R = np.concatenate([np.array(ts_mh["Recovered"]),     np.array(mh.R)])
    D = np.concatenate([np.array(ts_mh["Deceased"]),      np.array(mh.D)])

    fig = plt.figure()
    ax  = plt.gca()
    plt.suptitle("Maharashtra Migration Example (No Adaptive Controls; $R_0^{(0)} = " + str(round(mh.RR0, 3)) + "$)")
    plt.title("Lockdown Lifted: March 31st")
    s = ax.semilogy(S, linewidth = 1, label="Susceptible")
    i = ax.semilogy(I, linewidth = 1, label="Infectious" )
    d = ax.semilogy(D, linewidth = 1, label="Deaths"     )
    r = ax.semilogy(R, linewidth = 1, label="Recovered"  )
    ax.semilogy([len(ts_mh), len(ts_mh)], [0, 10e8], 'k--', alpha = 0.5, linewidth = 1)
    ax.set(xlabel = "# days since March 10", ylabel = "S/I/R/D")
    ax.label_outer()
    
    fig.legend([s, i, r, d], labels = ["S", "I", "R", "D"], loc = "center right", borderaxespad = 2)
    # plt.subplots_adjust(right=0.85)
    plt.show()
