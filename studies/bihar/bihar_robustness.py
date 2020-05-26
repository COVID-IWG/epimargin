from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics 
from tqdm import tqdm

import etl
from adaptive.estimators import rollingOLS
from adaptive.model import Model, ModelUnit
from adaptive.plots import gantt_chart, plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, fmt_params, weeks


def auc(model: Model, curve: str = "I") -> float:
    return metrics.auc(*list(zip(*enumerate(model.aggregate(curve)))))

def evaluate(models: Sequence[Tuple[Model]]):
    adaptive_dominant_trials = 0
    auc_scores = [[], [], []]
    for modelset in models:
        scores = [auc(model) for model in modelset]
        for (perf, s) in zip(auc_scores, scores):
            perf.append(s)
        if scores[-1] == min(scores):
            adaptive_dominant_trials += 1 

    return [np.mean(scores) for scores in auc_scores] + [float(adaptive_dominant_trials)/len(models)]

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
        total:          int   = 120*days,         # how long to run simulation
        eval_period:    int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling:   float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:           int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 9 day lockdown 
    model_A = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_A, 9*days, total, Rmw, Rvw, lockdown, migrations)

    # 9 day + 2 week lockdown 
    model_B = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_B, 39*days, total, Rmw, Rvw, lockdown, migrations)

    # 9 day lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, 9*days, total, lockdown, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

def estimate(district, ts, default = 1.5, window = 2, use_last = False):
    try:
        regressions = rollingOLS(etl.log_delta(ts), window = window, infectious_period = 1/gamma)[["R", "Intercept", "gradient", "gradient_stderr"]]
        if use_last:
            return next((x for x in regressions.R.iloc[-3:-1] if not np.isnan(x) and x > 0), default)
        return regressions
    except (ValueError, IndexError):
        return default

def gantt_seed(seed, note = ""):
    _, _, mc = run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, seed = seed) 
    gantt_chart(mc.gantt)\
        .title(f"Bihar: Example Adaptive Lockdown Mobility Regime Scenario {note if note else str(seed)}")\
        .show()

def project(p: pd.Series):
    t = (p.R - p.Intercept)/p.gradient
    return (max(0, p.R), max(0, p.Intercept + p.gradient*(t + 7)), max(0, p.Intercept + p.gradient*(t + 14)), np.sqrt(p.gradient_stderr))

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    
    gamma  = 0.2
    window = 2
    total_time = 110 * days 
    release_date = pd.to_datetime("June 1, 2020")
    lockdown_period = (release_date - pd.to_datetime("today")).days
    prevalence = 1

    state_cases    = etl.load_cases(data/"Bihar_Case_data_May18.csv")
    district_cases = etl.split_cases_by_district(state_cases)
    district_ts    = {district: etl.get_time_series(cases) for (district, cases) in district_cases.items()}
    R_mandatory    = {district: estimate(district, ts, use_last = True) for (district, ts) in district_ts.items()}
    districts, pops, migrations = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
    for district in districts:
        if district not in R_mandatory.keys():
            R_mandatory[district] = 1.5
    
    R_voluntary    = {district: 1.5*R for (district, R) in R_mandatory.items()}

    si, sf = 0, 1000

    results = []
    for beta_scaling in [1.0, 1.1, 1.25, 0.90, 0.75]:
        simulation_results = [ 
            run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, beta_scaling = beta_scaling, seed = seed)
            for seed in tqdm(range(si, sf))
        ]
        results.append([beta_scaling, gamma, prevalence] + evaluate(simulation_results)) 
    beta_scaling = 1
    for gamma in [0.1, 0.3, 0.4, 0.5]:
        simulation_results = [ 
            run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, beta_scaling = beta_scaling, seed = seed)
            for seed in tqdm(range(si, sf))
        ]
        results.append([beta_scaling, gamma, prevalence] + evaluate(simulation_results)) 

    gamma = 0.2
    for prevalence in [2, 4, 8]:
        prevalence_adj = {d: ts.copy() for (d, ts) in district_ts.items()}
        for (district, ts) in prevalence_adj.items():
            if "Hospitalized" in ts:
                ts["Hospitalized"] *= prevalence
        simulation_results = [ 
            run_policies(prevalence_adj, pops, districts, migrations, gamma, R_mandatory, R_voluntary, beta_scaling = beta_scaling, seed = seed)
            for seed in tqdm(range(si, sf))
        ]
        results.append([beta_scaling, gamma, prevalence] + evaluate(simulation_results)) 

    results_df = pd.DataFrame(results, columns = ["beta_scale", "gamma", "prevalence", "score_lockdown_short", "score_lockdown_long", "score_AC", "dominance"])
    results_df.to_csv(figs/"robustness.csv")