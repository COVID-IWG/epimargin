from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

def load_case_data(path: Path) -> pd.DataFrame: 
    df = pd.read_csv(path, usecols = ["date", "cases", "ward"])
    df["date"] = pd.to_datetime(df["date"], format="%b-%d") + pd.offsets.DateOffset(year=2020)
    df.ward = df.ward.apply(lambda w: w.split()[0])
    return df.set_index("date")

def log_delta(cases: pd.DataFrame) -> pd.DataFrame:
    ld = pd.DataFrame(np.log(cases.cases)).rename(columns = {"cases" : "logdelta"})
    ld["time"] = (ld.index - ld.index.min()).days
    return ld   

def load_migration_data(path: Path) -> Tuple[Sequence[str], np.matrix]:
    M = pd.read_csv(path).pivot(index = "O_ward", columns = "D_ward", values = "Num_passengers")
    P = np.matrix(M)
    P = P/P.sum(axis = 0)
    P[np.isnan(P)] = 0

    return ([name.strip() for name in M.index], P)

def load_population_data(path: Path) -> pd.Series:
    return pd.read_csv(path, usecols = ["pop"])["pop"]