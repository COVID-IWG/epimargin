from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

import etl
from adaptive.model import Model, ModelUnit
from adaptive.utils import *
from ward_model import gamma, get_R, run_policies


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


def main(N: int = 10_000):
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
        for seed in tqdm(range(N))
    ]

    results.append(["empirical beta, empirical gamma"] + evaluate(simulation_results))
    
    pd.DataFrame(
        data = results, 
        columns = ["parameters", "03may_avg_auc", "31may_avg_auc", "ac_avg_auc"]
    ).to_csv(sims/"simulation_results.csv")

if __name__ == "__main__":
    main()
