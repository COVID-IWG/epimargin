import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
from scipy.signal import convolve, filtfilt, iirnotch
from sklearn.linear_model import LinearRegression

from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import (get_time_series, load_statewise_data,
                                       state_name_lookup)
from adaptive.utils import cwd, mkdir

palette = [[0.8423298817793848, 0.8737404427964184, 0.7524954030731037], [0.5815252468131623, 0.7703468311289211, 0.5923205247665932], [0.35935359003014994, 0.6245622005326175, 0.554154071059354], [0.25744332683867743, 0.42368146872794976, 0.5191691971789514], [0.21392162678343224, 0.20848424698401846, 0.3660805512579508]]
sns.despine()
title_font = {}

def delay_adjust(confirmed, p_delay):
    "Adjust for empirical reporting delays, and additional adjustment for right-censoring"
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1], periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    cumulative_p_delay = p_delay.cumsum()

    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    # Calculate the new date range
    adj = onset / cumulative_p_delay
    adr = pd.date_range(end=onset.index[-1], periods=len(adj))
    adjusted = pd.Series(adj, index = adr)
    
    return adjusted

def plot_delay_dist(geo_name, delay_hist, cutoff = 100, show = False, filename = None):
    "Plot empirical reporting delay and fit negative exponential"
    delay_hist = delay_hist[delay_hist > 0]

    plt.bar(x = delay_hist.index, height = delay_hist.values, color = palette[-1])
    plt.xlim(-1, 120)
    plt.ylabel("percentage of cases\n", fontdict=label_font)
    plt.xlabel("\ndelay (days)", fontdict=label_font)
    ax = plt.gca()
    ax.yaxis.label.set_color("dimgray")
    ax.xaxis.label.set_color("dimgray")
    if filename: 
        plt.gcf().set_size_inches(11, 8)
        plt.savefig(filename, dpi = 600)
    if show: 
        plt.show()


def notch_filter(ts):
    "Implement notch filter with notches at {1/7, 2/7}"
    fs, f0, Q = 1, 1/7, 1
    b1, a1 = iirnotch(f0, Q, fs)
    b2, a2 = iirnotch(2*f0, 2*Q, fs)
    b = convolve(b1, b2)
    a = convolve(a1, a2)
    notched = pd.Series(filtfilt(b, a, ts))
    notched.index = ts.index
    return notched

root = cwd()
data = mkdir(root/"data")
figs = mkdir(root/"figs")

###########################################################
# download latest case data
download_data(data, 'state_wise_daily.csv')
df = load_statewise_data(data/"state_wise_daily.csv")
ts = get_time_series(df, "state")

###########################################################
# load delay data
api_diff = pd.read_csv(data/"daily_diff.csv", parse_dates=["status_change_date", "report_date"],  dayfirst=True)
delay = api_diff[(api_diff.current_status == "Hospitalized") & (api_diff.report_date > "2020-08-02")].copy()
delay = delay.drop(columns = [col for col in delay.columns if col.startswith("Unnamed")] + ["rowhash"])
delay["newhash"] = delay[["patient_number", "date_announced", "detected_district", "detected_state","current_status", "status_change_date", "num_cases"]].apply(lambda x: hash(tuple(x)), axis = 1)
delay = delay.drop_duplicates(subset=["newhash"], keep="first")
delay["delay"] = (delay.report_date - delay.status_change_date).dt.days
state_hist = delay[["detected_state", "num_cases", "delay"]].set_index(["detected_state", "delay"])["num_cases"].sum(level = [0, 1]).sort_index()
state_dist = state_hist/state_hist.sum(level = 0)

delay_hist = delay.groupby("delay")["num_cases"].sum()
delay_dist = delay_hist/delay_hist.sum()

###########################################################
# plot empirical delays at state and national levels
slopes = {}
slopes["TT"] = plot_delay_dist("all", delay_hist, show = True, filename = figs/"empirical_distribution_TT.png")
for state in ["Maharashtra"]:
    plt.figure()
    state_code = state_name_lookup.get(state, state)
    plot_delay_dist(state_code, 100*state_dist.loc[state], show = True, filename = figs/f"empirical_distribution_{state_code}.png")
    print(tikzplotlib.get_tikz_code())
    plt.close()
slopes["DDDN"] = slopes.pop("Dadra And Nagar Haveli And Daman And Diu")

plt.scatter(x = range(len(slopes)), y = sorted(slopes.values(), reverse=True), label = "all states")
plt.scatter(x = next(i for i in range(len(slopes)) if slope_labels[i] == "TT"), y = slopes["TT"], c='r', label = "national")
slope_labels = sorted(slopes.keys(), key=slopes.__getitem__, reverse=True)
plt.xticks(ticks = range(len(slopes)), labels = slope_labels)
plt.title("Exponential Fit Slope Coefficients for Empirical Delay Distributions", fontdict=title_font, loc="left")
plt.xlabel("\nstate", fontdict=label_font)
plt.ylabel("coefficient\n", fontdict=label_font)
plt.gca().yaxis.label.set_color("dimgray")
plt.gca().xaxis.label.set_color("dimgray")
plt.legend(prop={"family":"Helvetica Neue", "size": 12}, facecolor = "white")
plt.show()

###########################################################
# aggregated, delay-adjusted, notch-filtered data
confirmed = ts.Hospitalized.copy()
dndd = pd.DataFrame(confirmed["Dadra & Nagar Haveli"] + confirmed["Daman & Diu"])
dndd["state"] = "Dadra And Nagar Haveli And Daman And Diu"
dndd["status_change_date"] = dndd.index
dndd = dndd.set_index(["state", "status_change_date"])

state_notched = pd.concat([confirmed, dndd.Hospitalized])\
    .drop(["India", "Dadra & Nagar Haveli", "Daman & Diu"], axis = 0)\
    .groupby(level=0)\
    .apply(notch_filter)\
    .clip(0)

state_adj = {}
for state in [_ for _ in state_notched.index.get_level_values(0).unique() if _ not in ["Lakshadweep", "State Unassigned"]]:
    state_adj[state] = delay_adjust(state_notched[state], state_dist[state.replace("&", "And")])
agg_adj = sum(state_adj.values())

###########################################################
# plot MH example
mh_raw   = ts.loc["Maharashtra"].Hospitalized
mh_notch = notch_filter(ts.loc["Maharashtra"].Hospitalized)
mh_adj   = delay_adjust(ts.loc["Maharashtra"].Hospitalized, state_dist.loc["Maharashtra"])
mh_notch_adj = notch_filter(mh_adj)
mh_adj_notch = delay_adjust(mh_notch, state_dist.loc["Maharashtra"])

plt.plot(mh_raw,       label = "raw")
plt.plot(mh_notch,     label = "notch")
plt.plot(mh_adj,       label = "adj")
plt.plot(mh_notch_adj, label = "notch_adj")
plt.plot(mh_adj_notch, label = "adj_notch")
plt.legend(prop={"family":"Helvetica Neue", "size": 12}, facecolor = "white")

plt.xlabel("date\n", fontdict=label_font)
plt.ylabel("\nnumber cases", fontdict=label_font)
plt.gca().yaxis.label.set_color("dimgray")
plt.gca().xaxis.label.set_color("dimgray")
plt.xlim(left = pd.Timestamp("2020-01-01"))
plt.title(f"Case Timeseries Adjustments for Maharashtra", loc = "left", fontdict = title_font)
plt.show()

###########################################################
# plot all-India example
notched = notch_filter(ts.loc["India"].Hospitalized)

plt.plot([_.to_pydatetime() for _ in confirmed["India"].index], confirmed["India"].values, label = "raw case count API data", color = palette[-1], alpha = 0.5)
plt.plot([_.to_pydatetime() for _ in notched.index], notched.values, label = "notch-filtered case counts", color = palette[-1])

plt.legend()
plt.ylabel("cases")
plt.xlabel("date")
ax = plt.gca()

plt.xlim(left = pd.Timestamp("2020-06-01"), right = pd.Timestamp("2020-09-01"))
# plt.ylim(top=90000)
plt.gcf().autofmt_xdate()
# plt.semilogy()
print(tikzplotlib.get_tikz_code())
plt.show()
