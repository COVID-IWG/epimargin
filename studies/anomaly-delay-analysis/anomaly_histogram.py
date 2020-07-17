from typing import Optional
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft
from scipy.signal import blackman
from tqdm import tqdm

from adaptive.estimators import gamma_prior
from adaptive.etl.covid19india import (download_data, get_time_series,
                                       load_statewise_data, state_name_lookup)
from adaptive.smoothing import convolution
from adaptive.utils import cwd

simplefilter("ignore")
sns.set(palette="bright", font="Inconsolata")
sns.despine()

title_font = {"size": 20, "family": "Libre Franklin", "fontweight": "400"}
label_font = {"size": 16, "family": "Libre Franklin", "fontweight": "300"}

def plot_average_change(ts: pd.DataFrame, label: str = "", filename: Optional[str] = None, show: bool = False):
    ts.groupby("dow")["delta_I"].agg(np.mean).reset_index().plot.bar(x = "dow", y = "delta_I", width = 0.1) 
    ax = plt.gca()
    ax.get_legend().remove()
    plt.xlabel("\nDay of Week", fontdict=label_font)
    plt.ylabel("Average ΔI",  fontdict=label_font)
    ax.xaxis.label.set_alpha(0.75)
    ax.yaxis.label.set_alpha(0.75)
    xlabel_locs, _ = plt.xticks()
    plt.xticks(xlabel_locs, ["M", "Tu", "W", "Th", "F", "Sa", "Su"], rotation = 0)
    plt.title(f"Average Change in Reported Infections by Day of Week {label}", loc="left", fontdict=title_font)
    if filename or show:
        plt.gcf().set_size_inches(11, 8)
    if filename:
        plt.savefig(filename, dpi=600)

    if show:
        plt.show()
    else: 
        plt.clf()

def anomaly_histogram(anomaly_dates, label: str = "", filename: Optional[str] = None, show: bool = False):
    plt.hist([_.dayofweek for _ in anomaly_dates], bins = range(8), align = "left", rwidth=0.5) 
    plt.xlabel("\nDay of Week", fontdict=label_font)
    plt.ylabel("Number of Anomalies", fontdict=label_font)
    ax = plt.gca()
    ax.xaxis.label.set_alpha(0.75)
    ax.yaxis.label.set_alpha(0.75)
    xlabel_locs, _ = plt.xticks()
    plt.xticks(xlabel_locs, ["", "M", "Tu", "W", "Th", "F", "Sa", "Su"], rotation = 0)
    plt.title(f"Anomalies by Day of Week {label}", loc="left", fontdict=title_font)
    if filename or show:
        plt.gcf().set_size_inches(11, 8)
    if filename:
        plt.savefig(filename, dpi=600)

    if show:
        plt.show()
    else: 
        plt.clf()

def spectrum(ts: pd.DataFrame, label: str):
    # y = ts["Hospitalized"].fillna(0).diff().values[1:]
    y = ts["Hospitalized"].fillna(0).values
    T = 1 # daily period 
    N = len(y)
    yf = fft(y)
    w = blackman(N)
    ywf = fft(y*w)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf[3:N//2], 2/N * np.abs(ywf[3:N//2]), ".", alpha = 0.7, label=label)

root = cwd()
data = root/"data"
figs = root/"figs"

download_data(data, 'state_wise_daily.csv')
state_df = load_statewise_data(data/"state_wise_daily.csv")
natl_time_series = get_time_series(state_df)
time_series      = get_time_series(state_df, 'state')

# is there chunking in reporting?
print("checking average infection differentials...")
time_series["delta_I"] = time_series.groupby(level=0)['Hospitalized'].diff()
time_series["dow"] = time_series.index.get_level_values(1).dayofweek
plot_average_change(time_series, "(All India)", filename=figs/"avg_delta_I_DoW_India.png")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    plot_average_change(time_series.loc[state], f"({state})", filename=figs/f"avg_delta_I_DoW_{state}.png")

# are anomalies falling on certain days?
print("checking anomalies...")
smoothing = 5 
(*_, anomaly_dates) = gamma_prior(natl_time_series["Hospitalized"].iloc[:-1], CI = 0.95, smoothing = convolution(window = smoothing)) 
anomaly_histogram(anomaly_dates, "(All India)", filename=figs/"anomaly_DoW_hist_India.png")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    (*_, anomaly_dates) = gamma_prior(time_series.loc[state]["Hospitalized"].iloc[:-1], CI = 0.95, smoothing = convolution(window = smoothing)) 
    anomaly_histogram(anomaly_dates, f"({state})", filename=figs/f"anomaly_DoW_hist_{state}.png")

print("estimating spectral densities...")
# what does the aggregate spectral density look like?
spectrum(natl_time_series, "All India")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    spectrum(time_series.loc[state], state)
plt.axvline(1/7,   ls='--', color="black", alpha = 0.5)
plt.axvline(1/3.5, ls='--', color="black", alpha = 0.5)
plt.axvline(1/30,  ls='--', color="black", alpha = 0.5)
plt.text(1/7,   200, r" $\nu$ = 1/7 days")
plt.text(1/3.5, 200, r" $\nu$ = 2/7 days")
plt.text(1/30,  200, r" $\nu$ = 1/30 days")
plt.xlabel(r"$\nu$")
plt.ylabel(r"$|\mathcal{F}(\nu)|$")
plt.legend(loc="upper right", fontsize = "x-small")
# plt.semilogy()
plt.title(f"Spectral Density Estimates for Reported Infection Counts", loc="left", fontdict=title_font)
plt.gcf().set_size_inches(11, 8)
plt.show()
