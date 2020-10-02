from itertools import cycle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patheffects import Stroke, Normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import *

from .model import Model


# from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if not float_list:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('ACRt', segmentdata=cdict, N=256)
    mpl.cm.register_cmap("ACRt", cmp)
    return cmp

_ = plt 
sns.despine()
mpl.rcParams["savefig.dpi"] = 300

# palettes

## Rt
### core plot
BLK    = "#292f36"
BLK_CI = "#aeb7c2"

### stoplight 
RED = "#D63231"
YLW = "#FD8B5A"
GRN = "#38AE66"
sm = mpl.cm.ScalarMappable(
    norm = mpl.colors.Normalize(vmin = 0, vmax = 3), 
    cmap = get_continuous_cmap([GRN, YLW, RED, RED], [0, 0.8, 0.9, 1])
)

## new cases
OBS_BLK     = BLK
CASE_BLU    = "#335970"
ANOMALY_BLU = "#4092A0"
PRED_PURPLE = "#554B68"

## policy simulations 
SIM_PALETTE = ["#437034", "#7D4343", "#43587D", "#7D4370"]


# typography
title_font = {"size": 28, "family": "Helvetica Neue", "fontweight": "500"}
label_font = {"size": 20, "family": "Helvetica Neue", "fontweight": "500"}
note_font  = {"size": 14, "family": "Helvetica Neue", "fontweight": "500"}
ticks_font = {"family": "Inconsolata"}
sns.set(style = "whitegrid", palette = "bright", font = "Helvetica Neue")

plt.rcParams['mathtext.default'] = 'regular'
DATE_FMT = mdates.DateFormatter('%d %b')


# simple wrapper over plt to help chain commands
class PlotDevice():
    def __init__(self, fig: Optional[mpl.figure.Figure] = None):
        self.figure = fig if fig else plt.gcf()
        
    def xlabel(self, xl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", label_font)
        plt.xlabel(xl, **kwargs)
        plt.gca().xaxis.label.set_color("dimgray")
        return self 

    def ylabel(self, yl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", label_font)
        plt.ylabel(yl, **kwargs)
        plt.gca().yaxis.label.set_color("dimgray")
        return self 

    def title(self, text: str, **kwargs):
        try:
            kwargs["x"]  = kwargs.get("x", self.figure.get_axes()[0].get_position().bounds[0])
        except IndexError:
            kwargs["x"]  = kwargs.get("x", plt.gca().get_position().bounds[0])
        kwargs["ha"] = kwargs.get("ha", "left")
        kwargs["va"] = kwargs.get("va", "top")
        kwargs["fontsize"] = kwargs.get("fontsize", title_font["size"])
        kwargs["fontdict"] = kwargs.get("fontdict", title_font)
        plt.suptitle(text, **kwargs)
        return self 
    
    def annotate(self, text: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", note_font)
        kwargs["loc"] = kwargs.get("loc", "left")
        plt.title(text, **kwargs)
        return self
    
    def size(self, w, h):
        self.figure.set_size_inches(w, h)
        return self

    def save(self, filename: Path, **kwargs):
        kwargs["transparent"] = kwargs.get("transparent", str(filename).endswith("svg"))
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
            "format": mpl.ticker.FuncFormatter(lambda x, pos: {0.5:"voluntary", 1:"cautionary", 2:"partial", 2.5:"restricted"}[x]), 
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

def simulations(
    simulation_results: Sequence[Tuple[Model]], 
    labels: Sequence[str], 
    historical: Optional[pd.Series] = None, 
    historical_label: str = "Empirical Case Data", 
    curve: str = "delta_T", 
    smoothing: Optional[np.ndarray] = None) -> PlotDevice:

    aggregates = [tuple(model.aggregate(curve) for model in model_set) for model_set in simulation_results]

    policy_outcomes = list(zip(*aggregates))

    num_sims   = len(simulation_results)
    total_time = len(policy_outcomes[0][0])

    ranges = [{"max": [], "min": [], "mdn": [], "avg": []} for _ in range(len(policy_outcomes))]

    for (i, policy) in enumerate(policy_outcomes):
        for t in range(total_time):
            curve_sorted = sorted([curve[t] for curve in policy])
            ranges[i]["min"].append(curve_sorted[0])
            ranges[i]["max"].append(curve_sorted[-1])
            ranges[i]["mdn"].append(curve_sorted[num_sims//2])
            ranges[i]["avg"].append(np.mean(curve_sorted))

    legends = []
    legend_labels  = []
    if historical is not None:
        p, = plt.plot(historical.index, historical, 'k-', alpha = 0.8, zorder = 10)
        t  = [historical.index.max() + pd.Timedelta(days = n) for n in range(total_time)]
        legends.append(p)
        legend_labels.append(historical_label)
    else:
        t = list(range(total_time))

    if smoothing is not None:
        p = plt.plot([pd.Timestamp(t) for t in smoothing[:, 0]], smoothing[:, 1], 'k-', linewidth = 1)
        legends.append(p)
        legend_labels.append("smoothed_data")
        
    for (rng, label, color) in zip(ranges, labels, SIM_PALETTE):
        p, = plt.plot(t, rng["avg"], color = color, linewidth = 2)
        f  = plt.fill_between(t, rng["min"], rng["max"], color = color, alpha = 0.2)
        legends.append((p, f))
        legend_labels.append(label)
    
    plt.gca().xaxis.set_major_formatter(DATE_FMT)
    plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    plt.legend(legends, legend_labels, prop = dict(size = 20), handlelength = 1, framealpha = 1)

    plt.xlim(left = historical.index[0], right = t[-1])
    
    return plt.PlotDevice()

def Rt(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin = 0.5, ymax = 3):
    try: 
        dates = [_.to_pydatetime() for _ in dates]
    except AttributeError:
        pass 
    CI_marker  = plt.fill_between(dates, RR_CI_lower, RR_CI_upper, color = BLK_CI, alpha = 0.5)
    Rt_marker, = plt.plot(dates, RR_pred, color = BLK, linewidth = 2, zorder = 5, solid_capstyle = "butt")
    plt.plot([dates[0], dates[0]], [2.5, ymax], color = RED, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
    plt.plot([dates[0], dates[0]], [1,    2.5], color = YLW, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
    plt.plot([dates[0], dates[0]], [ymin,   1], color = GRN, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
    plt.plot([dates[0], dates[0]], [ymin, ymax], color = "white", linewidth = 10, alpha = 1, solid_capstyle="butt", zorder = 9)
    plt.hlines(1, xmin=dates[0], xmax=dates[-1], zorder = 11, color = "black", linestyles = "dotted")
    plt.ylim(ymin, ymax)
    plt.xlim(left=dates[0], right=dates[-1])
    plt.legend([(CI_marker, Rt_marker)], [f"Estimated $R_t$ ({100*CI}% CI)"], prop = {'size': 16}, framealpha = 1, handlelength = 1)
    plt.gca().xaxis.set_major_formatter(DATE_FMT)
    plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    return PlotDevice()

def daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, predictions = None, pred_CI_upper = None, pred_CI_lower = None):
    observed_marker,   = plt.plot(dates[-len(new_cases_ts):], new_cases_ts, color = OBS_BLK, linewidth = 2, zorder = 8)
    anomalies_marker,  = plt.plot(anomaly_dates, anomalies, marker="o", mfc = "none", mec=ANOMALY_BLU, lw = 0)
    expected_marker,   = plt.plot(dates[-len(T_pred):], T_pred, color = CASE_BLU, marker=".", zorder = 10, lw = 0)
    expected_CI_marker = plt.fill_between(dates, T_CI_lower, T_CI_upper, label = f"", facecolor = CASE_BLU, alpha = 0.4)
    legends = [observed_marker, (expected_CI_marker, expected_marker)]
    labels  = ["observed cases (smoothed)", f"expected cases ({100*CI}% CI)"]
    (_, top) = plt.ylim()
    if predictions:
        t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(predictions))]
        end = t_pred[-1]
        predicted_marker,   = plt.plot(t_pred, predictions, color = PRED_PURPLE, marker = ".", zorder = 9, lw = 0)
        predicted_CI_marker = plt.fill_between(t_pred, pred_CI_lower, pred_CI_upper, facecolor = PRED_PURPLE, alpha = 0.4)
        (_, top) = plt.ylim()
        plt.vlines(dates[-1], ymin = 0, ymax = 2*top, color = "black", linestyles = "dotted")
        legends += [(predicted_CI_marker, predicted_marker)]

        labels  += [f"predicted cases ({100*CI}% CI)"]
    else: 
        end = dates[-1]
    plt.ylim(bottom = 0, top = top)
    legends += [anomalies_marker]
    labels  += ["anomalies"]
    xlim(left = dates[0], right = end)
    plt.legend(legends, labels, prop = {'size': 16}, framealpha = 1, handlelength = 1, loc = "upper left")
    plt.gca().xaxis.set_major_formatter(DATE_FMT)
    plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    return PlotDevice()

def choropleth(gdf, label_fn = lambda _: "", Rt_col = "Rt", Rt_proj_col = "Rt_proj", titles = ["Current $R_t$", "Projected $R_t$ (1 Week)"], arrangement = (1, 2), label_kwargs = {}, mappable = sm):
    gdf["pt"] = gdf["geometry"].centroid
    fig, (ax1, ax2) = plt.subplots(*arrangement)
    for (ax, title, col) in zip((ax1, ax2), titles, (Rt_col, Rt_proj_col)):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontdict=label_font) 
        gdf.plot(color=[mappable.to_rgba(_) for _ in gdf[col]], ax = ax, edgecolors="black", linewidth=0.5, missing_kwds = {"color": "dimgray", "edgecolor": "white"})
    if label_fn is not None:
        for (_, row) in gdf.iterrows():
            label = label_fn(row)
            Rt_c, Rt_p = round(row[Rt_col], 2), round(row[Rt_proj_col], 2)
            a1 = ax1.annotate(s=f"{label}{Rt_c}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = note_font["family"], color="white", **label_kwargs)
            a2 = ax2.annotate(s=f"{label}{Rt_p}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = note_font["family"], color="white", **label_kwargs)
            a1.set_path_effects([Stroke(linewidth = 2, foreground = "black"), Normal()])
            a2.set_path_effects([Stroke(linewidth = 2, foreground = "black"), Normal()])
    cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    cb = fig.colorbar(mappable = mappable, orientation = "vertical", cax = cbar_ax)
    cbar_ax.set_title("$R_t$", fontdict = note_font)
    
    return PlotDevice(fig)

def choropleth_v(*args, **kwargs):
    kwargs["arrangement"] = (2, 1)
    return choropleth(*args, **kwargs)

choropleth.horizontal = choropleth
choropleth.vertical   = choropleth_v
