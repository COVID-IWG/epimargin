import csv
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import cm
from numpy import log as ln
from scipy.stats import gamma as gamma_distribution
from scipy.stats import nbinom, poisson
from tqdm import tqdm

from adaptive.estimators import rollingOLS as run_regressions
from adaptive.model import Model, ModelUnit, gravity_matrix
from adaptive.plots import plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks
from etl import download_data, get_time_series, load_all_data


def smooth(y, smoothing):
    box = np.ones(smoothing)/smoothing
    y_smooth = np.convolve(y, box, mode='same')

    # local smoothing 
    if len(y_smooth) > 5:
        y_smooth[-2] = (y[-4] + y[-3] + y[-2])/3
        y_smooth[-1] = (y[-3] + y[-2] + y[-1])/3

    return y_smooth

def gamma_prior(ts, inf_period = 5, smoothing = 5):
    if "Hospitalized" not in ts:
        return [], []
    dconfirmed = np.diff(ts["Hospitalized"])
    dconfirmed[dconfirmed < 0] = 0
    try:
        ts_h_smooth = smooth(dconfirmed, smoothing)
    except ValueError:
        print("smoothing error, dconfirmed:", list(dconfirmed))
        return [], []
    total_cases = np.cumsum(ts_h_smooth)

    alpha, beta = 3, 2
    valpha, vbeta = [], []

    pred = []
    pstdM = []
    pstdm = []
    xx = []
    new_cases  = []

    predR = []
    pstRRM = []
    pstRRm = []

    anomalyday = []
    anomalypred = []

    for i in (range(2, len(total_cases))):
        # print(i)
        delta_alpha = total_cases[i]   - total_cases[i-1]
        delta_beta  = total_cases[i-1] - total_cases[i-2]

        alpha += delta_alpha
        beta  += delta_beta
        valpha.append(alpha)
        vbeta.append(beta)

        RR_estimated = max(0, 1 + inf_period * ln(gamma_distribution.stats(a=alpha, scale=1/beta, moments='m')))
        RR_upper     = max(0, 1 + inf_period * ln(gamma_distribution.ppf(0.95, a=alpha, scale=1/beta)))
        RR_lower     = max(0, 1 + inf_period * ln(gamma_distribution.ppf(0.05, a=alpha, scale=1/beta)))
        predR.append(RR_estimated)
        pstRRM.append(RR_upper)
        pstRRm.append(RR_lower)

        if delta_alpha == 0 or delta_beta == 0:
            pred.append(0.)
            pstdM.append(10.)
            pstdm.append(0.)
            new_cases.append(0.)
        if delta_alpha > 0 and delta_beta > 0:
            new_cases.append(delta_alpha)
            r, p = alpha, beta/(delta_beta + beta)
            
            mean = nbinom.stats(r, p, moments='m')
            pred.append(mean) # the expected value of new cases
            testciM = nbinom.ppf(0.95, r, p) # these are the boundaries of the 99% confidence interval  for new cases
            pstdM.append(testciM)
            testcim = nbinom.ppf(0.05, r, p)
            pstdm.append(testcim)

            new_p = p
            new_r = r 
            flag = 0 

            iters = 0

            while not (testcim < delta_alpha < testciM):
                # print("anomaly")
                # print(testcim, delta_alpha, testciM)
                # print()
                if (flag == 0):
                    anomalypred.append(delta_alpha)
                    anomalyday.append(ts.index[i])
                nnp = 0.95 * new_p
                new_r = new_r*(nnp/new_p)*( (1.-new_p)/(1.-nnp) )
                new_p = nnp 
                testciM = nbinom.ppf(0.95, new_r, new_p)
                testcim = nbinom.ppf(0.05, new_r, new_p)

                flag = 1
                iters += 1
                if iters > 100 and testciM == 0:
                    print("error - no convergence", testcim, delta_alpha, testciM)
                    return ([], [])
            else: 
                if (flag == 1):
                    alpha = new_r
                    beta  = new_p/(1 - new_p) * delta_beta

                    testciM = nbinom.ppf(0.95, new_r, new_p)
                    testcim = nbinom.ppf(0.05, new_r, new_p)

                    #pstdM=pstdM[:-1] # remove last element and replace by expanded CI for New Cases
                    #pstdm=pstdm[:-1]  # This (commented) in  order to show anomalies, but on
                    #pstdM.append(testciM) # in the parameter update, uncomment and it will plot the actual updated CI
                    #pstdm.append(testcim)

                    # annealing leaves the RR mean unchanged, but we need to adjust its widened CI:
                    testRRM = max(0, 1.+inf_period*ln( gamma_distribution.ppf(0.99, a=alpha, scale=1./beta) ))# these are the boundaries of the 99% confidence interval  for new cases
                    testRRm = max(0, 1.+inf_period*ln( gamma_distribution.ppf(0.01, a=alpha, scale=1./beta) ))
                    
                    pstRRM=pstRRM[:-1] # remove last element and replace by expanded CI for RRest
                    pstRRm=pstRRm[:-1]
                    pstRRM.append(testRRM)
                    pstRRm.append(testRRm)
    ts["Rt"]          = [np.nan, np.nan, np.nan] + predR
    ts["Rt_upper_CI"] = [np.nan, np.nan, np.nan] + pstRRM
    ts["Rt_lower_CI"] = [np.nan, np.nan, np.nan] + pstRRm
    ts["total_I"]     = [np.nan] + list(total_cases)
    linfit = sm.OLS(ts.iloc[-5:].Rt, sm.add_constant(sm.add_constant(ts.iloc[-5:].time - ts.iloc[-1].time, prepend=False))).fit()
    k1, k0 = linfit.params
    sigma = linfit.bse["time"]
    ts_proj = pd.DataFrame(
        [(ts.date.iloc[-1] + pd.Timedelta(days = n), max(0, k0 + k1*n), max(0, k0 + k1*n - 0.96*sigma), max(0, k0 + k1*n + 0.96*sigma)) for n in range(1, 6)], 
        columns = ["date", "Rt_proj", "Rt_proj_lower_CI", "Rt_proj_upper_CI"])
    # plt.plot(ts.iloc[3:].index, smooth(predR, 10))
    # plt.fill_between(ts.iloc[3:].index, smooth(pstRRM, 10), smooth(pstRRm, 10),color='gray', alpha=0.15, label="95% Confidence Interval")
    return (ts, ts_proj)

if __name__ == "__main__":
    root = cwd()
    data = root/"data"

    # model details 
    gamma      = 0.2
    prevalence = 1

    states = [
        'Andhra Pradesh',
        'Assam',
        'Bihar',
        'Chandigarh',
        'Chhattisgarh',
        'Dadra and Nagar Haveli and Daman and Diu',
        'Delhi',
        'Goa',
        'Gujarat',
        'Haryana',
        'Himachal Pradesh',
        'Jammu and Kashmir',
        'Jharkhand',
        'Karnataka',
        'Kerala',
        'Ladakh',
        'Madhya Pradesh',
        'Maharashtra',
        'Manipur',
        'Meghalaya',
        'Odisha',
        'Puducherry',
        'Punjab',
        'Rajasthan',
        'Sikkim',
        'Tamil Nadu',
        'Telangana',
        'Tripura',
        'Uttar Pradesh',
        'Uttarakhand',
        'West Bengal'
    ]
    
    # use gravity matrix for states after 2001 census 
    new_state_data_paths = { 
        "Telangana": (data/"telangana.json", data/"telangana_pop.csv")
    }

    # define data versions for api files
    paths = { "v3": ["raw_data1.csv", "raw_data2.csv"],
              "v4": ["raw_data3.csv", "raw_data4.csv",
                     "raw_data5.csv", "raw_data6.csv"] } 

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    # run rolling regressions on historical national case data 
    dfn = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )

    # migration_matrices = district_migration_matrices(states, data/"Migration Matrix - District.csv")

    data_recency = str(dfn["date_announced"].max()).split()[0]
    tsn = get_time_series(dfn)
    estn, projn = gamma_prior(tsn)
    estn.to_csv(data/"website_natl_est.csv")
    projn.to_csv(data/"website_natl_rtproj.csv")

    # disaggregate down to states
    dfs = {state: dfn[dfn["detected_state"] == state] for state in states}
    tss = {state: get_time_series(cases) for (state, cases) in dfs.items()}
    for (state, ts) in tss.items():
        # if state not in {"Arunachal Pradesh", "Chhattisgarh", "Himachal Pradesh"}:
        # if state in ss:
        print(state)
        try: 
            ests, projs = gamma_prior(ts)
            if len(ests) > 0:
                ests.to_csv(data/(f"website_{state}_est.csv"))
            if len(projs) > 0:
                projs.to_csv(data/(f"website_{state}_rtproj.csv"))
        except KeyError as e:
            print(e)
            pass


    for state in states: 
        print(f"[{state}]")
        df_state = dfs[state]
        districts = list(df_state["detected_district"].unique())
        dfd  = {district: df_state[df_state["detected_district"] == district] for district in districts}
        tsd  = {district: get_time_series(cases) for (district, cases) in dfd.items()}
        # estd = {district: gamma_prior(ts) for (district, ts) in tsd.items()}
        Rvals = []
        for district in districts:
            print(" ", district)
            try:
                est, proj = gamma_prior(tsd[district])
            except Exception as e:
                print(e)
                est, proj = [], []
            Rvals.append([district, est.iloc[-1].Rt if len(est) > 0 else np.nan, max(0, proj.iloc[-1].Rt_proj) if len(proj) > 0 else np.nan])
        pd.DataFrame(Rvals, columns=["district", "Rt", "Rt_proj"]).to_csv(data/(f"website_{state}_districts.csv"))
