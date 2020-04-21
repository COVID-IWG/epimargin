#!python3 

"""code to create logarithmic growth rate plots for internationally-compiled data"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import seaborn

def parse(count_str: str):
    counts = map(int, count_str.split("-"))
    counts += [0] * (4 - len(counts))
    return (counts[0], counts[2], counts[3])

# assuming analysis for data structure from virus.csv 
def load_data(datapath: Path) -> pd.DataFrame: 
    df = pd.read_csv(datapath)
    df[["confirmed", "recovered", "dead"]] = None
        # skiprows    = 1, # supply fixed header in order to deal with Google Sheets export issues 
        # names       = columns, 
        # usecols     = (lambda _: _ not in drop_cols) if reduced else None,
        # dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        # parse_dates = ["date announced", "status change date"])

def assume_missing_0(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0

# calculate daily totals and growth rate
def get_time_series(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby(["status change date", "current status"])["patient number"].count().unstack().fillna(0)
    totals["date"]     = totals.index
    totals["time"]     = (totals["date"] - totals["date"].min()).dt.days
    totals["logdelta"] = np.log(assume_missing_0(totals, "Hospitalized") - assume_missing_0(totals, "Recovered") -  assume_missing_0(totals, "Deceased"))
    return totals

def run_regressions(totals: pd.DataFrame, window: int = 3, infectious_period: float = 4.5) -> pd.DataFrame:
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

def critical_day(growthrates: pd.DataFrame) -> Tuple[int, int, int, int]:
    # extrapolate growth rate into the future
    predrates = growthrates.iloc[-5:].copy()
    predrates["days"] -= predrates["days"].min()
    pred = sm.OLS.from_formula("gradient ~ days", data = predrates).fit()
    pred_intercept, pred_gradient, pred_se = *pred.params, pred.bse[1]
    days_to_critical  = int(-pred_intercept/pred_gradient)
    days_to_criticalM = int(-pred_intercept/(pred_gradient + 2 * pred_se))
    days_to_criticalm = int(-pred_intercept/(pred_gradient - 2 * pred_se))

    return (len(growthrates.index), days_to_critical, days_to_criticalm, days_to_criticalM)

def plot_rates(totals: pd.DataFrame, growthrates: pd.DataFrame, state: str, label: str, note: str, show_plots: bool = False):
    output = Path(__file__).parent/"plots"

    # figure: log delta vs time
    fig, ax = plt.subplots()
    totals.plot(y = "logdelta", ax = ax, label = "log(confirmed - recovered - dead)")
    plt.xlabel("Date")
    plt.ylabel("Daily Net Cases")
    plt.title(f"Cases over Time ({state})")
    plt.tight_layout()
    plt.savefig(output/f"cases_over_time_{label}{note}.png", dpi = 600)

    # figure: extrapolation
    # t0 = growthrates.iloc[-5:].days.max()
    # t = np.arange(t0, t0 + max(days_to_criticalm, days_to_criticalM) + 1)
    # pred_lower = pred_intercept + t * (pred_gradient - 2 * pred_se)
    # pred_upper = pred_intercept + t * (pred_gradient + 2 * pred_se)
    
    fig, ax = plt.subplots()
    # plt.fill_between(t, pred_upper, pred_lower, alpha = 0.3)
    
    plt.plot(growthrates.days, growthrates.gradient)
    plt.fill_between(growthrates.days, growthrates.egrowthratem, growthrates.egrowthrateM, alpha = 0.3)
    plt.xlabel("Days of Outbreak")
    plt.ylabel("Growth Rate")
    plt.title(state)
    plt.tight_layout()
    plt.savefig(output/f"extrapolation_{label}{note}.png", dpi = 600)

    # figure: reproductive rate vs critical level 
    fig, ax = plt.subplots()
    plt.plot(growthrates.date, growthrates.R)
    plt.fill_between(growthrates.date, growthrates.Rm, growthrates.RM, alpha = 0.3)
    plt.xlabel("Date")
    plt.ylabel("Reproductive Rate")
    plt.title(state)
    plt.tight_layout()
    plt.savefig(output/f"reproductive_rate_{label}{note}.png", dpi = 600)

    if show_plots: 
        plt.show()
    plt.close("all")

def run_analysis(df: pd.DataFrame, state: str = "all", note: str = "", show_plots: bool = False) -> Tuple[int, int, int, int]:
    # filter data as needed and set up filename components
    if state and state.replace(" ", "").lower() not in ("all", "allstate", "allstates"):
        df = df[df["detected state"] == state]
        label = state.replace(" ", "_").lower()
    else: 
        state = "All States"
        label = "allstates"
    
    if len(df) < 3:
        warnings.warn(f"Insufficient data for {state}")
        return 0, None, None, None
    
    note = '_' + note if note else ''

    totals      = get_time_series(df)
    growthrates = run_regressions(totals)
    plot_rates(totals, growthrates, state, label, note, show_plots)

    return critical_day(growthrates)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seaborn.set_style('darkgrid')
    
    root = Path(__file__).parent
    df = load_data(root/"india_case_data_resave.csv", reduced = True)

    states = ["all"]
    critical_days = []

    for state in states:
        n, dtc, dtcm, dtcM = run_analysis(df)
        critical_days.append([state, n, dtc, dtcm, dtcM]) 
    
    pd.DataFrame(critical_days, columns = ["state", "n", "dct", "dctm", "dctM"])\
        .set_index("state")\
        .to_csv(root/"state_dct.csv")
