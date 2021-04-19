from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

import etl
from epimargin.model import Model, ModelUnit
from epimargin.utils import *
from ward_model import gamma, get_R, run_policies


def auc(model: Model, curve: str = "I") -> float:
    return metrics.auc(*list(zip(*enumerate(model.aggregate(curve)))))

def auc_ts(ts, curve: str = "I") -> float:
    return metrics.auc(*list(zip(*enumerate(ts))))

def parse_runs(runs):
    for series_tuple in runs[["0", "1", "2"]].itertuples(index = False):
        yield tuple(auc_ts(map(int, series[1:-1].split(", "))) for series in series_tuple)
    # [auc_ts([int(n.replace("[", "").replace("]", "")) for n in curve.split(", ")]) for curve in runs[["0", "1", "2"]].itertuples(index = False)]

def evaluate(models: Sequence[Tuple[Model]]):
    adaptive_dominant_trials = 0
    auc_scores = [[], [], []]
    for modelset in models:
        scores = [auc(model) for model in modelset]
        for (perf, s) in zip(auc_scores, scores):
            perf.append(s)
        if scores[-1] == min(scores):
            adaptive_dominant_trials += 1 

    return (
        [np.mean(scores) for scores in auc_scores] + [float(adaptive_dominant_trials)/len(models)],
        auc_scores
    )

def auc_dist():
    runs = pd.concat([pd.read_csv(f"auc_runs_{i}.csv") for i in range(15)])
    scoresA, scoresB, scoresC = zip(*list(parse_runs(runs)))
    
    # binsA = np.linspace(min(scoresA), max(scoresA), 1000)
    # binsB = np.linspace(min(scoresB), max(scoresB), 1000)
    # binsC = np.linspace(min(scoresC), max(scoresC), 1000)

    binmin = int(min(*scoresA, *scoresB, *scoresC))
    binmax = int(max(*scoresA, *scoresB, *scoresC))
    bins = range(binmin, binmax, 250000)

    plt.hist(scoresA, bins = bins, log=True, label='03 May Release')
    plt.hist(scoresB, bins = bins, log=True, label='31 May Release')
    plt.hist(scoresC, bins = bins, log=True, label='Adaptive Release')
    plt.legend(loc='upper right')
    plt.ylim(1, 10e4)
    plt.suptitle("Distribution of AUC Scores")
    plt.title("15000 Policy Simulations for Mumbai Wards")
    plt.xlabel("AUC")
    plt.ticklabel_format(axis = "x", style= "plain")
    plt.show()


if __name__ == "__main__":
    N = 100
    import sys
    index = int(sys.argv[1]) - 1
    root = cwd()
    data = root/"data"
    sims = root/"sims"

    all_cases         = etl.load_case_data(data/"mumbai_wards_30Apr.csv")
    population_data   = etl.load_population_data(data/"ward_data_Mumbai_empt_slums.csv")
    wards, migrations = etl.load_migration_data(data/"Ward_rly_matrix_Mumbai.csv")

    ward_cases = {ward: all_cases[all_cases.ward == ward] for ward in wards}
    Rmw, Rvw = get_R(ward_cases, gamma)

    results = []
    simulation_results = [
        run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, seed = seed) 
        for seed in tqdm(range(index*N, (index + 1)*N))
    ]

    evaluation, auc_scores = evaluate(simulation_results)
    results.append(["empirical beta / empirical gamma"] + evaluation)
    
    pd.DataFrame(
        [[model.aggregate("I") for model in models] for models in simulation_results]
    ).to_csv(sims/f"auc_runs_{index}.csv")

    pd.DataFrame(
        data = results, 
        columns = ["parameters", "03may_avg_auc", "31may_avg_auc", "ac_avg_auc", "dominance"]
    ).to_csv(sims/f"simulation_results_{index}.csv")
