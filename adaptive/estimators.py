from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gamma as Gamma
from scipy.stats import nbinom
from statsmodels.regression.rolling import RollingOLS

from .utils import days


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
    growthrates["date"]         = growthrates.index.get_level_values('status_change_date')
    growthrates["days"]         = totals.time

    return growthrates

# the Gamma distribution is the Bayesian prior and posterior for a Poisson process 
def gamma_prior(
        infection_ts: pd.DataFrame, 
        smoothing: Callable,
        alpha: float = 3.0,                  # shape 
        beta:  float = 2.0,                  # rate
        CI:    float = 0.95,                 # confidence interval 
        infectious_period: int = 5*days       # inf period = 1/gamma  
    ):
    dates = infection_ts.iloc[1:].index
    infection_ts = infection_ts.copy(deep = True)
    infection_ts[infection_ts < 0] = 0
    daily_cases_raw = np.diff(infection_ts).clip(min = 0)
    daily_cases = smoothing(daily_cases_raw)

    total_cases = np.cumsum(daily_cases)

    v_alpha, v_beta = [], []

    RR_pred, RR_CI_upper, RR_CI_lower = [], [], []

    T_pred, T_CI_upper, T_CI_lower = [], [], []

    new_cases_ts = []

    anomalies     = []
    anomaly_dates = []

    for i in range(2, len(total_cases)):
        new_cases     = total_cases[i]   - total_cases[i-1]
        old_new_cases = total_cases[i-1] - total_cases[i-2]

        alpha += new_cases
        beta  += old_new_cases
        v_alpha.append(alpha)
        v_beta.append(beta)

        RR_est   = max(0, 1 + infectious_period*np.log(Gamma.mean(     a = alpha, scale = 1/beta)))
        RR_upper = max(0, 1 + infectious_period*np.log(Gamma.ppf(CI,   a = alpha, scale = 1/beta)))
        RR_lower = max(0, 1 + infectious_period*np.log(Gamma.ppf(1-CI, a = alpha, scale = 1/beta)))
        RR_pred.append(RR_est)
        RR_CI_upper.append(RR_upper)
        RR_CI_lower.append(RR_lower)

        if (new_cases == 0 or old_new_cases == 0):
            T_pred.append(0)
            T_CI_upper.append(10) # <- where does this come from?
            T_CI_lower.append(0)
            new_cases_ts.append(0)

        if (new_cases > 0 and old_new_cases > 0): # <- why is this the entry condition for the nbinom?
            new_cases_ts.append(new_cases)

            r, p = alpha, beta/(old_new_cases + beta)
            T_pred.append(nbinom.mean(r, p))
            T_upper = nbinom.ppf(CI,   r, p)
            T_lower = nbinom.ppf(1-CI, r, p)
            T_CI_upper.append(T_upper)
            T_CI_lower.append(T_lower)

            _np = p
            _nr = r 
            flag = 0 # why?
            while not (T_lower < new_cases < T_upper):
                if (flag == 0):
                    anomalies.append(new_cases)
                    anomaly_dates.append(dates[i])
                
                # nnp = 0.95 *_np # <- where does this come from 
                _nr = variance_shift * _nr * ((1-_np)/(1-variance_shift*_np) )
                _np = variance_shift * _np 
                T_upper = nbinom.ppf(CI,   _nr, _np)
                T_lower = nbinom.ppf(1-CI, _nr, _np)
                T_lower, T_upper = sorted((T_lower, T_upper))
                flag = 1
            else:
                if (flag == 1):
                    alpha = _nr # update distribution on R with new parameters that enclose the anomaly 
                    beta = _np/(1-_np) * old_new_cases

                    T_upper = nbinom.ppf(CI,   _nr, _np)
                    T_lower = nbinom.ppf(1-CI, _nr, _np)

                    # annealing leaves the RR mean unchanged, but we need to adjust its widened CI
                    RR_upper = max(0, 1 + infectious_period * np.log(Gamma.ppf(CI    , a = alpha, scale = 1/beta)))
                    RR_lower = max(0, 1 + infectious_period * np.log(Gamma.ppf(1 - CI, a = alpha, scale = 1/beta)))

                    # replace latest CI time series entries with adjusted CI 
                    RR_CI_upper[-1] = RR_upper
                    RR_CI_lower[-1] = RR_lower
    return (
        dates[2:], 
        RR_pred, RR_CI_upper, RR_CI_lower, 
        T_pred, T_CI_upper, T_CI_lower, 
        total_cases, new_cases_ts, 
        anomalies, anomaly_dates
    )
