from typing import Optional
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft, ifft
from scipy.signal import blackman, periodogram, spectrogram, stft, welch, iirnotch, freqz, convolve, filtfilt
from scipy.stats import chi2
from tqdm import tqdm

from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import (download_data, get_time_series,
                                       load_statewise_data)
from adaptive.smoothing import convolution
from adaptive.utils import cwd, weeks as week

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

def spectrum(ts: pd.Series, label: str):
    y = ts.fillna(0).values
    T = 1 # daily period 
    N = len(y)
    yf = fft(y)
    w = blackman(N)
    ywf = fft(y*w)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf[1:N//2], 2/N * np.abs(ywf[1:N//2]), ".", alpha = 0.7, label=label)

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
(*_, anomaly_dates) = analytical_MPVS(natl_time_series["Hospitalized"].iloc[:-1], CI = 0.95, smoothing = convolution(window = smoothing)) 
anomaly_histogram(anomaly_dates, "(All India)", filename=figs/"anomaly_DoW_hist_India.png")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    (*_, anomaly_dates) = analytical_MPVS(time_series.loc[state]["Hospitalized"].iloc[:-1], CI = 0.95, smoothing = convolution(window = smoothing)) 
    anomaly_histogram(anomaly_dates, f"({state})", filename=figs/f"anomaly_DoW_hist_{state}.png")

print("estimating spectral densities...")
# what does the aggregate spectral density look like?
spectrum(natl_time_series["Hospitalized"], "All India")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    spectrum(time_series.loc[state]["Hospitalized"], state)
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


print("estimating noise floor on periodogram...")
f, Pxx = welch(natl_time_series["Hospitalized"], scaling="density")
CI_upper = 2*Pxx/chi2.ppf(0.975, df=2)
CI_lower = 2*Pxx/chi2.ppf(0.025, df=2)
plt.semilogy(f, Pxx, label = "spectrum coefficients", color="blue")
plt.fill_between(f, CI_lower, CI_upper, alpha = 0.2, color="blue")
plt.axvline(1/7,  ls='--', color="black", alpha = 0.5)
plt.axvline(2/7,  ls='--', color="black", alpha = 0.5)
plt.axvline(1/124, ls='--', color="black", alpha = 0.5)
plt.text(1/7,  1e11, r" $\nu$ = 1/7 days")
plt.text(2/7,  1e11, r" $\nu$ = 2/7 days")
plt.text(1/124, 1e11, r" $\nu$ = 1/N days")
plt.title(f"Spectral Density Confidence Interval for Reported India Infection Counts (N = 124)", loc="left", fontdict=title_font)
plt.tight_layout()
# plt.gcf().set_size_inches(11, 8)
plt.show()

print("estimating periodogram (welch's method)...")
# what does the aggregate spectral density look like?
plt.plot(*welch(natl_time_series["Hospitalized"], scaling="density"), label = "All India")
for state in tqdm(time_series.index.get_level_values(0).unique()):
    plt.plot(*welch(time_series.loc[state]["Hospitalized"], scaling="density"), label=state)
plt.axvline(1/7,  ls='--', color="black", alpha = 0.5)
plt.axvline(2/7,  ls='--', color="black", alpha = 0.5)
plt.axvline(1/30, ls='--', color="black", alpha = 0.5)
plt.text(1/7,  200, r" $\nu$ = 1/7 days")
plt.text(2/7,  200, r" $\nu$ = 2/7 days")
plt.text(1/30, 200, r" $\nu$ = 1/30 days")
# plt.xlabel(r"$\nu$")
# plt.ylabel(r"$|\mathcal{F}(\nu)|$")
plt.legend(loc="upper right", fontsize = "x-small")
plt.semilogy()
plt.title(f"Welch Periodograms for Reported Infection Counts", loc="left", fontdict=title_font)
plt.gcf().set_size_inches(11, 8)
plt.show()

print("estimating STFT...")
f, t, Zxx = stft(natl_time_series.Hospitalized, nperseg=12)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.show()

y = natl_time_series.Hospitalized.values
# signal design 
fs, f0, Q = 1, 1/7, 1
b1, a1 = signal.iirnotch(f0, Q, fs)
b2, a2 = signal.iirnotch(2*f0, 2*Q, fs)
# Frequency response
b = convolve(b1, b2)
a = convolve(a1, a2)

yff = filtfilt(b, a, y)
plt.plot(y,   label = "original")
plt.plot(yff, label = "notch-filtered")
plt.legend()
plt.title("Notch-Filtered vs Original Signal", loc="left", fontdict=title_font)
plt.tight_layout()
plt.show()

freq, h = signal.freqz(b, a, fs=fs)
# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].grid()
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].grid()
plt.suptitle("Bode Plot for Weekly+Semiweekly Notch Filter", fontdict=title_font)
plt.tight_layout()
plt.show()

# causal vs non-causal smoothing 
window = lambda n: np.ones(n)/n
plt.plot(yff, label = "notch-filtered")
plt.plot(convolve(yff, window(10))[:-10], label = "10 day window")
plt.plot(convolve(yff, window(20))[:-20], label = "20 day window")
plt.legend()
plt.title("Causal Smoothing", loc="left", fontdict=title_font)
plt.tight_layout()
plt.show()

plt.plot(yff, label = "notch-filtered")
plt.plot(convolve(yff, window(10), mode ="same"), label = "10 day window")
plt.plot(convolve(yff, window(20), mode ="same"), label = "20 day window")
plt.legend()
plt.title("Non-Causal Smoothing", loc="left", fontdict=title_font)
plt.tight_layout()
plt.show()

plt.scatter(range(len(y)), y, color = "purple", label = "original", s=2)
plt.plot(yff, label = "notch-filtered", color="black")
plt.plot(np.concatenate([yff, yff[:-21:-1]]), linestyle = "dotted", color = "black", label = "time-reversed padding")
plt.plot(convolve(np.concatenate([yff, yff[:-6:-1]]),  window(5),  mode ="same")[:-5], label = "5  day window (padded)", alpha = 0.5)
plt.plot(convolve(np.concatenate([yff, yff[:-8:-1]]),  window(7),  mode ="same")[:-7], label = "7  day window (padded)", alpha = 0.5)
plt.plot(convolve(np.concatenate([yff, yff[:-11:-1]]), window(10), mode ="same")[:-11], label = "10 day window (padded)", alpha = 0.5)
plt.plot(convolve(np.concatenate([yff, yff[:-21:-1]]), window(20), mode ="same")[:-21], label = "20 day window (padded)", alpha = 0.5)
plt.legend()
plt.title("Non-Causal Smoothing, with Time-Reversed Padding", loc="left", fontdict=title_font)
plt.tight_layout()
plt.show()