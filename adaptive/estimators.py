from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.rolling import RollingOLS

from .utils import assume_missing_0


# estimators 
def rollingOLS(totals: pd.DataFrame, window: int = 3, infectious_period: float = 4.5) -> pd.DataFrame:
    # run rolling regressions and get parameters
    model   = RollingOLS.from_formula(formula = "logdelta ~ time", window = window, data = totals)
    rolling = model.fit(method = "lstsq")
    
    growthrates = rolling.params.join(rolling.bse, rsuffix="_stderr")
    growthrates["rsq"] = rolling.rsquared
    growthrates.rename(lambda s: s.replace("time", "gradient").replace("const", "intercept"), axis = 1, inplace = True)

    # calculate growth rates
    growthrates["egrowthrateM"] = growthrates.gradient + 2 * growthrates.gradient_stderr
    growthrates["egrowthratem"] = growthrates.gradient - 2 * growthrates.gradient_stderr
    growthrates["R"]            = growthrates.gradient * infectious_period + 1
    growthrates["RM"]           = growthrates.gradient + 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["Rm"]           = growthrates.gradient - 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["date"]         = growthrates.index
    growthrates["days"]         = totals.time

    return growthrates

# data prep 
def log_delta(time_series: pd.DataFrame, I: Optional[str] = None, R: Optional[str] = None, D: Optional[str] = None, cases: Optional[str] = "cases", smoothing: bool = True):
    if I and R and D: 
        time_series[cases] = assume_missing_0(time_series, I) - assume_missing_0(time_series, R) - assume_missing_0(time_series, D)
    if smoothing:
        time_series[cases] = lowess(time_series[cases], time_series.index)[:, 1] 
    log_delta = pd.DataFrame(np.log(time_series.cases)).rename(columns = {"cases" : "logdelta"})
    log_delta["time"] = (log_delta.index - log_delta.index.min()).days
    return log_delta


def log_delta(time_series: pd.DataFrame, I: Optional[str] = None, R: Optional[str] = None, D: Optional[str] = None, cases: Optional[str] = "cases", smoothing: bool = True):
    if I and R and D: 
        time_series[cases] = time_series[I] - time_series[R] - time_series[D]
    if smoothing:
        time_series[cases] = lowess(time_series[cases], time_series.index)[:, 1] 
    log_delta = pd.DataFrame(np.log(time_series.cases)).rename(columns = {"cases" : "logdelta"})
    log_delta["time"] = (log_delta.index - log_delta.index.min()).days
    return log_delta
