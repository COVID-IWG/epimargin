from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_case_data(path: Path) -> pd.DataFrame: 
    return pd.read_csv(path, 
        usecols = ["date", "cases", "ward"],
        parse_dates = ["date"]
    ).set_index("date")

def load_migration_data(path: Path) -> Tuple[Sequence[str], np.matrix]:
    M = pd.read_csv(path).pivot(index = "D_ward", columns = "O_ward", values = "Num_passengers")
    P = np.matrix(M)
    P = P/P.sum(axis = 0)
    P[np.isnan(P)] = 0

    return (list(M.index), P)

def load_population_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols = ["ward", "pop"])