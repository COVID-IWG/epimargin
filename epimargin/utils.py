import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

days     = 1
weeks    = 7 
years    = 365
percent  = 1/100
annually = 1/years
million  = 1e6

def cwd() -> Path:
    argv0 = sys.argv[0]
    if argv0.endswith("ipython"):
        return Path(".").resolve()
    return Path(argv0).resolve().parent
        
def fmt_params(**kwargs) -> str:
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for (k, v) in kwargs.items())

def assume_missing_0(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0

def mkdir(p: Path, exist_ok: bool = True) -> Path:
    p.mkdir(exist_ok=exist_ok)
    return p

def setup(**kwargs) -> Tuple[Path]:
    root = cwd()
    if len(sys.argv) > 2:
        parser = argparse.ArgumentParser()
        parser.add_argument("--level", type=str)
        flags = parser.parse_args()
        kwargs["level"] = flags.level
    logging.basicConfig(**kwargs)
    logging.getLogger('flat_table').addFilter(lambda _: 0)
    return (mkdir(root/"data"), mkdir(root/"figs"))

def fillna(array):
    return np.nan_to_num(array, nan = 0, posinf = 0, neginf = 0)

def normalize(array, axis = 0):
    return fillna(array/array.sum(axis = axis)[:, None])
