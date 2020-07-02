from pathlib import Path
from typing import Optional, Sequence, Tuple
from itertools import cycle

import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model import Model

sns.set(style = "whitegrid", palette = "bright", font = "Fira Code")
sns.despine()


# simple wrapper over plt to help chain commands
class PlotDevice():
    def __init__(self, fig: Optional[mlp.figure.Figure] = None):
        self.figure = fig if fig else plt.gcf()
        
    def xlabel(self, xl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
        plt.xlabel(xl, **kwargs)
        return self 

    def ylabel(self, yl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
        plt.ylabel(yl, **kwargs)
        return self 

    def title(self, text: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", {"size": 20, "family": "Fira Sans", "fontweight": "500"})
        kwargs["loc"]      = kwargs.get("loc", "left")
        plt.title(text, **kwargs)
        return self 
    
    def annotate(self, note: str, **kwargs):
        kwargs["xy"] = kwargs.get("xy", (0.05, 0.05))
        kwargs["xycoords"] = kwargs.get("xycoords", "figure fraction")
        kwargs["size"] = kwargs.get("size", 8)
        plt.annotate(note, **kwargs)
        return self
    
    def size(self, w, h):
        self.figure.set_size_inches(w, h)
        return self

    def save(self, filename: Path, **kwargs):
        if "transparent" not in kwargs.keys():
            kwargs["transparent"] = str(filename).endswith("svg")
        plt.savefig(filename, **kwargs)
        return self 

    def adjust(self, **kwargs):
        plt.subplots_adjust(**kwargs)
        return self 

    def show(self, **kwargs):
        plt.show(**kwargs)
        return self 

# plot all 4 curves
def plot_SIRD(model: Model) -> PlotDevice:
    fig, axes = plt.subplots(1, 4, sharex = True, sharey = True)
    t = list(range(len(model[0].RR)))
    for (ax, model) in zip(axes.flat, model.units):
        s = ax.semilogy(t, model.S, alpha=0.75, label="Susceptibles")
        i = ax.semilogy(t, model.I, alpha=0.75, label="Infectious", )
        d = ax.semilogy(t, model.D, alpha=0.75, label="Deaths",     )
        r = ax.semilogy(t, model.R, alpha=0.75, label="Recovered",  )
        ax.label_outer()
    
    fig.legend([s, i, r, d], labels = ["S", "I", "R", "D"], loc="center right", borderaxespad=0.1)
    return PlotDevice(fig)

# plot a single curve 
def plot_curve(models: Sequence[Model], labels: Sequence[str], curve: str = "I"):
    fig = plt.figure()
    for (model, label) in zip(models, labels):
        plt.semilogy(model.aggregate(curve), label = label, figure = fig)
    plt.legend() 
    plt.tight_layout()
    return PlotDevice(fig)

def gantt_chart(gantt_data, start_date: Optional[str] = None, show_cbar = True):
    gantt_df = pd.DataFrame(gantt_data, columns = ["district", "day", "beta", "R"])
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    if start_date:
        start_timestamp = pd.to_datetime(start_date)
        dates = [start_timestamp + pd.Timedelta(days = n) for n in gantt_df.day.unique()]
        xticklabels = [str(xl.day) + " " + xl.month_name()[:3] for xl in dates]
        xlabel = "Date"
    else:
        xticklabels = sorted(gantt_df.day.unique())
        xlabel = "Days Since Beginning of Adaptive Control"
    ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        annot = gantt_pv["R"], annot_kws={"size": 8},
        cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"],
        cbar = show_cbar,
        yticklabels = gantt_df["district"].unique(),
        xticklabels = xticklabels,
        cbar_kws = {
            "ticks":[0.5, 1, 2, 2.5], 
            "label": "Mobility", 
            "format": mlp.ticker.FuncFormatter(lambda x, pos: {0.5:"voluntary", 1:"cautionary", 2:"partial", 2.5:"restricted"}[x]), 
            "orientation": "horizontal", 
            "aspect": 50, 
            "drawedges": True,
            "fraction": 0.05,
            "pad": 0.10, 
            "shrink": 0.5
        }
    )
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.10, bottom=0.10)
    plt.xlabel(xlabel, {"size": 20, "family": "Fira Sans", "fontweight": "500"})
    plt.ylabel(None)

    return PlotDevice()

def plot_simulation_range(
    simulation_results: Sequence[Tuple[Model]], 
    labels: Sequence[str], 
    historical: Optional[pd.Series] = None, 
    historical_label: str = "Empirical Case Data", 
    curve: str = "I", 
    smoothing: Optional[np.ndarray] = None) -> PlotDevice:

    aggregates = [tuple(model.aggregate(curve) for model in model_set) for model_set in simulation_results]

    policy_outcomes = list(zip(*aggregates))

    num_sims   = len(simulation_results)
    total_time = len(policy_outcomes[0][0])

    ranges = [{"max": [], "min": [], "mdn": []} for _ in range(len(policy_outcomes))]

    for (i, policy) in enumerate(policy_outcomes):
        for t in range(total_time):
            curve_sorted = sorted([curve[t] for curve in policy])
            ranges[i]["min"].append(curve_sorted[0])
            ranges[i]["max"].append(curve_sorted[-1])
            ranges[i]["mdn"].append(curve_sorted[num_sims//2])

    if historical is not None:
        plt.semilogy(historical.index, historical, 'k.', label = historical_label, alpha = 0.8, zorder = 10)
        t = [historical.index.max() + pd.Timedelta(days = n) for n in range(total_time)]
    else:
        t = list(range(total_time))

    if smoothing is not None:
        plt.semilogy([pd.Timestamp(t) for t in smoothing[:, 0]], smoothing[:, 1], 'k-', label = "LOESS smoothed data", linewidth = 1)
        
    for (rng, label) in zip(ranges, labels):
        plt.semilogy(t, rng["mdn"], label = label, linewidth = 2)
        plt.fill_between(t, rng["min"], rng["max"], alpha = 0.2)
    
    
    plt.gca().format_xdata = mdates.DateFormatter('%m-%d')
    plt.legend()
    
    return PlotDevice()

def plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin = 0, ymax = 4):
    fig = plt.figure()
    plt.plot(dates, RR_pred, label = "Estimated $R_t$", color = "darkorchid")
    plt.fill_between(dates, RR_CI_lower, RR_CI_upper, label = f"{100*CI}% CI", color = "darkorchid", alpha = 0.3)
    # plt.ylim(ymin, ymax)
    plt.legend()
    return PlotDevice(fig)

def plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI):
    fig = plt.figure()
    plt.scatter(dates[-len(new_cases_ts):], new_cases_ts, color = "mediumpurple", marker=".", label="Observed Cases (smoothed)")
    plt.scatter(anomaly_dates, anomalies, label = "Anomalies", marker="o", color="crimson", facecolors = 'none')
    plt.plot(dates[-len(T_pred):], T_pred, label = "Expected Cases", color = "darkcyan")
    plt.fill_between(dates, T_CI_lower, T_CI_upper, label = f"{100*CI}% CI", facecolor = "gray", alpha = 0.3)
    plt.legend()
    return PlotDevice(fig)

