import logging
from typing import Callable, Optional, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import gamma as Gamma
from scipy.stats import nbinom
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant

from .utils import days

logger = logging.getLogger(__name__)

def rollingOLS(totals: pd.DataFrame, window: int = 3, infectious_period: float = 4.5) -> pd.DataFrame:
    """ legacy rolling regression-based implementation of Bettencourt/Ribeiro method """
    # run rolling regressions and get parameters
    model   = RollingOLS.from_formula(formula = "logdelta ~ time", window = window, data = totals)
    rolling = model.fit(method = "lstsq")
    
    growthrates = pd.DataFrame(rolling.params).join(rolling.bse, rsuffix="_stderr")
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

def analytical_MPVS(
        infection_ts: pd.DataFrame, 
        smoothing: Callable,
        alpha: float = 3.0,                # shape 
        beta:  float = 2.0,                # rate
        CI:    float = 0.95,               # confidence interval 
        infectious_period: int = 5*days,   # inf period = 1/gamma,
        variance_shift: float = 0.99,      # how much to scale variance parameters by when anomaly detected 
        totals: bool = True                # are these case totals or daily new cases?
    ):
    """Estimates Rt ~ Gamma(alpha, 1/beta), and implements an analytical expression for a mean-preserving variance increase whenever case counts fall outside the CI defined by a negative binomial distribution"""
    # infection_ts = infection_ts.copy(deep = True)
    dates = infection_ts.index
    if totals:
        # daily_cases = np.diff(infection_ts.clip(lower = 0)).clip(min = 0) # infection_ts clipped because COVID19India API does weird stuff
        daily_cases = infection_ts.clip(lower = 0).diff().clip(lower = 0).iloc[1:]
    else: 
        daily_cases = infection_ts 
    total_cases = np.cumsum(smoothing(np.squeeze(daily_cases)))

    v_alpha, v_beta = [], []

    Rt_pred, Rt_CI_upper, Rt_CI_lower = [], [], []

    T_pred, T_CI_upper, T_CI_lower = [], [], []

    new_cases_ts = []

    anomalies     = []
    anomaly_dates = []

    for i in range(2, len(total_cases)):
        new_cases     = max(0, total_cases[i]   - total_cases[i-1])
        old_new_cases = max(0, total_cases[i-1] - total_cases[i-2])

        alpha += new_cases
        beta  += old_new_cases
        v_alpha.append(alpha)
        v_beta.append(beta)

        Rt_est   = max(0, 1 + infectious_period*np.log(Gamma.mean(     a = alpha, scale = 1/beta)))
        Rt_upper = max(0, 1 + infectious_period*np.log(Gamma.ppf(CI,   a = alpha, scale = 1/beta)))
        Rt_lower = max(0, 1 + infectious_period*np.log(Gamma.ppf(1-CI, a = alpha, scale = 1/beta)))
        Rt_pred.append(Rt_est)
        Rt_CI_upper.append(Rt_upper)
        Rt_CI_lower.append(Rt_lower)

        if (new_cases == 0 or old_new_cases == 0):
            if new_cases == 0:
                logger.debug("new_cases at time %s: 0", i)
            if old_new_cases == 0:
                logger.debug("old_new_cases at time %s: 0", i)
            T_pred.append(0)
            T_CI_upper.append(10) # <- where does this come from?
            T_CI_lower.append(0)
            new_cases_ts.append(0)

        if (new_cases > 0 and old_new_cases > 0):
            new_cases_ts.append(new_cases)

            r, p = alpha, beta/(old_new_cases + beta)
            T_pred.append(nbinom.mean(r, p))
            T_upper = nbinom.ppf(CI,   r, p)
            T_lower = nbinom.ppf(1-CI, r, p)
            T_CI_upper.append(T_upper)
            T_CI_lower.append(T_lower)

            _np = p
            _nr = r 
            anomaly_noted = False
            counter = 0
            while not (T_lower < new_cases < T_upper):
                if not anomaly_noted:
                    anomalies.append(new_cases)
                    anomaly_dates.append(dates[i])
                
                # logger.debug("anomaly identified at time %s: %s < %s < %s, r: %s, p: %s, annealing iteration: %s", i, T_lower, new_cases, T_upper, _nr, _np, counter+1)
                # nnp = 0.95 *_np # <- where does this come from 
                _nr = variance_shift * _nr * ((1-_np)/(1-variance_shift*_np) )
                _np = variance_shift * _np 
                T_upper = nbinom.ppf(CI,   _nr, _np)
                T_lower = nbinom.ppf(1-CI, _nr, _np)
                T_lower, T_upper = sorted((T_lower, T_upper))
                if T_lower == T_upper == 0:
                    T_upper = 1
                    logger.debug("CI collapse, setting T_upper -> 1")
                anomaly_noted = True

                counter += 1
                if counter >= 10000:
                    raise ValueError("Number of iterations exceeded")
            else:
                if anomaly_noted:
                    alpha = _nr # update distribution on R with new parameters that enclose the anomaly 
                    beta = _np/(1-_np) * old_new_cases

                    T_pred[-1] = nbinom.mean(_nr, _np)
                    T_CI_lower[-1] = nbinom.ppf(CI,   _nr, _np)
                    T_CI_upper[-1] = nbinom.ppf(1-CI, _nr, _np)

                    # annealing leaves the RR mean unchanged, but we need to adjust its widened CI
                    Rt_upper = max(0, 1 + infectious_period * np.log(Gamma.ppf(CI    , a = alpha, scale = 1/beta)))
                    Rt_lower = max(0, 1 + infectious_period * np.log(Gamma.ppf(1 - CI, a = alpha, scale = 1/beta)))

                    # replace latest CI time series entries with adjusted CI 
                    Rt_CI_upper[-1] = Rt_upper
                    Rt_CI_lower[-1] = Rt_lower
    return (
        dates[2:], 
        Rt_pred, Rt_CI_upper, Rt_CI_lower, 
        T_pred, T_CI_upper, T_CI_lower, 
        total_cases, new_cases_ts, 
        anomalies, anomaly_dates
    )

def parametric_scheme_mcmc(daily_cases, CI = 0.95, gamma = 0.2, chains = 4, tune = 1000, draws = 1000, **kwargs):
    """ Implements the Bettencourt/Soman parametric scheme via MCMC sampling """
    if isinstance(daily_cases, (pd.DataFrame, pd.Series)):
        case_values = daily_cases.values
    else: 
        case_values = np.array(daily_cases)
    with pm.Model() as mcmc_model:
        # lag new case counts
        dT_lag0 = case_values[1:]
        dT_lag1 = case_values[:-1]
        n = len(dT_lag0)

        dT = pm.Poisson("dT", mu = dT_lag0, shape = (n,))
        bt = pm.Gamma("bt", alpha = dT_lag0.cumsum(), beta = 0.0001 + dT_lag1.cumsum(), shape = (n,))
        Rt = pm.Deterministic("Rt", 1 + pm.math.log(bt)/gamma)
    
        trace = pm.sample(model = mcmc_model, chains = chains, tune = tune, draws = draws, cores = 1, **kwargs)
        return (mcmc_model, trace, pm.summary(trace, hdi_prob = CI))

def branching_random_walk(daily_cases, CI = 0.95, gamma = 0.2, chains = 4, tune = 1000, draws = 1000, **kwargs):
    """ estimate Rt using a random walk for branch parameter, adapted from old Rt.live code """
    if isinstance(daily_cases, (pd.DataFrame, pd.Series)):
        case_values = daily_cases.values
    else: 
        case_values = np.array(daily_cases)
    with pm.Model() as mcmc_model:
        # lag new case counts
        dT_lag0 = case_values[1:]
        dT_lag1 = case_values[:-1]
        n = len(dT_lag0)
        
        # Random walk magnitude
        step_size = pm.HalfNormal('step_size', sigma = 0.03)
        theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
        theta_raw_steps = pm.Normal('theta_raw_steps', shape = n - 1) * step_size
        theta_raw = tt.concatenate([[theta_raw_init], theta_raw_steps])
        theta = pm.Deterministic('theta', theta_raw.cumsum())

        Rt = pm.Deterministic("Rt", 1 + theta/gamma)
        expected_cases = pm.Poisson('dT', mu = dT_lag1 * pm.math.exp(theta), observed = dT_lag0)
    
        trace = pm.sample(model = mcmc_model, chains = chains, tune = tune, draws = draws, cores = 1, **kwargs)
        return (mcmc_model, trace, pm.summary(trace, hdi_prob = CI))

def linear_projection(dates, R_values, smoothing, period = 7*days):
    """ return 7-day linear projection """
    julian_dates = [_.to_julian_date() for _ in dates[-smoothing//2:None]]
    return OLS(
        R_values[-smoothing//2:None], 
        add_constant(julian_dates)
    )\
    .fit()\
    .predict([1, julian_dates[-1] + period])[0]
