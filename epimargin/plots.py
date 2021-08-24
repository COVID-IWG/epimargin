import datetime
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
from matplotlib.patheffects import Normal, Stroke
from matplotlib.pyplot import *

from .models import NetworkedSIR


def normalize_dates(dates):
    try: 
        return [_.to_pydatetime().date() for _ in dates]
    except AttributeError:
        return dates

_ = plt # make mpl package available in epimargin.plots

# default settings
sns.despine()
mpl.rcParams["savefig.dpi"]     = 300
mpl.rcParams["xtick.labelsize"] = "large"
mpl.rcParams["ytick.labelsize"] = "large"
mpl.rcParams["svg.fonttype"]    = "none"

# palettes
## Rt
### core plot
BLK    = "#292f36"
BLK_CI = "#aeb7c2"

### stoplight 
RED = "#D63231"
YLW = "#FD8B5A"
GRN = "#38AE66"

## new cases
OBS_BLK     = BLK
CASE_BLU    = "#335970"
ANOMALY_BLU = "#4092A0"
ANOMALY_RED = "#D63231"
PRED_PURPLE = "#554B68"

## policy simulations 
SIM_PALETTE = ["#437034", "#7D4343", "#43587D", "#7D4370"]

# typography
def rebuild_font_cache():
    import matplotlib.font_manager
    matplotlib.font_manager._rebuild()

# container class for different theme
Aesthetics = namedtuple(
    "Aesthetics", 
    ["title", "label", "note", "ticks", "style", "palette", "accent", "despine", "framealpha", "handlelength"]
)

twitter_settings = Aesthetics(
    title   = {"size": 28, "family": "Overpass", "weight": "bold"},
    label   = {"size": 20, "family": "Overpass", "weight": "regular"},
    note    = {"size": 14, "family": "Overpass", "weight": "regular"},
    ticks   = {"size": 12, "family": "Overpass", "weight": "regular"},
    style   = "white",
    accent  = "dimgrey",
    palette = "bright",
    despine = True,
    framealpha = 0,
    handlelength = 0.5
)

substack_settings = Aesthetics(
    title   = {"size": 28, "family": "SF Pro Display", "weight": "medium"},
    label   = {"size": 20, "family": "SF Pro Display", "weight": "light"},
    note    = {"size": 14, "family": "SF Pro Display", "weight": "light"},
    ticks   = {"size": 12, "family": "Spectral"},
    style   = "white",
    accent  = "#8C8475",
    palette = "bright",
    despine = True,
    framealpha = 0,
    handlelength = 0.5
)

theme = default_settings = Aesthetics(
    title   = {"size": 28, "family": "Helvetica Neue", "weight": "regular"},
    label   = {"size": 20, "family": "Helvetica Neue", "weight": "regular"},
    note    = {"size": 14, "family": "Helvetica Neue", "weight": "regular"},
    ticks   = {"size": 10, "family": "Helvetica Neue"},
    style   = "whitegrid",
    palette = "bright",
    accent  = "dimgrey",
    despine = False,
    framealpha = 1,
    handlelength = 1
)

minimal_settings = Aesthetics(
    title   = {"size": 28, "family": "Helvetica Neue", "weight": "regular"},
    label   = {"size": 20, "family": "Helvetica Neue", "weight": "regular"},
    note    = {"size": 14, "family": "Helvetica Neue", "weight": "regular"},
    ticks   = {"size": 12, "family": "Helvetica Neue"},
    style   = "white",
    palette = "bright",
    accent  = "dimgrey",
    despine = True,
    framealpha = 0,
    handlelength = 0.5
)

plt.rcParams['mathtext.default'] = 'regular'
DATE_FMT = mdates.DateFormatter('%d %b')
bY_FMT   = mdates.DateFormatter('%b %Y')

def set_theme(name):
    global theme
    if   name == "twitter":
        theme = twitter_settings
    elif name == "substack":
        theme = substack_settings
    elif name == "minimal":
        theme = minimal_settings
    else: # default 
        theme = default_settings
    sns.set(style = theme.style, palette = theme.palette, font = theme.ticks["family"])
    mpl.rcParams.update({"font.size": 22})
    if theme.despine:
        plt.rc("axes.spines", top = False, right = False)
    return theme

set_theme("default")


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


# DEFAULT COLOR MAPPING
default_cmap = get_continuous_cmap([GRN, YLW, RED, RED], [0, 0.8, 0.9, 1])
def get_cmap(vmin = 0, vmax = 3, cmap = default_cmap):
    return mpl.cm.ScalarMappable(
        norm = mpl.colors.Normalize(vmin, vmax), 
        cmap = cmap
    )

sm = get_cmap()

def set_tick_size(size: int):
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)

# simple wrapper over plt to help chain commands
class PlotDevice():
    def __init__(self, fig: Optional[mpl.figure.Figure] = None):
        self.figure = fig if fig else plt.gcf()
        if theme.despine:
            sns.despine(top = True, right = True)

    def axis_labels(self, x, y, enforce_spacing = True, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", theme.label)
        if enforce_spacing and not x.startswith("\n"):
            x = "\n" + x 
        if enforce_spacing and not y.endswith("\n"):
            y = y + "\n"
        return self.xlabel(x, **kwargs).ylabel(y, **kwargs)

    def xlabel(self, xl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", theme.label)
        plt.xlabel(xl, **kwargs)
        plt.gca().xaxis.label.set_color("dimgray")
        return self 

    def ylabel(self, yl: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", theme.label)
        plt.ylabel(yl, **kwargs)
        plt.gca().yaxis.label.set_color("dimgray")
        return self 

    # stack title/subtitle vertically
    def title(self, text: str, **kwargs):
        try:
            kwargs["x"]  = kwargs.get("x", self.figure.get_axes()[0].get_position().bounds[0])
        except IndexError:
            kwargs["x"]  = kwargs.get("x", plt.gca().get_position().bounds[0])
        kwargs["ha"] = kwargs.get("ha", "left")
        kwargs["va"] = kwargs.get("va", "top")
        kwargs["fontsize"]   = kwargs.get("fontsize", theme.title["size"])
        kwargs["fontdict"]   = kwargs.get("fontdict", theme.title)
        kwargs["fontweight"] = kwargs.get("fontweight", theme.title["weight"])
        plt.suptitle(text, **kwargs)
        return self 
    
    def annotate(self, text: str, **kwargs):
        kwargs["fontdict"] = kwargs.get("fontdict", theme.note)
        kwargs["loc"] = kwargs.get("loc", "left")
        plt.title(text, **kwargs)
        return self

    # stack title/subtitle horizontally 
    def l_title(self, text: str, **kwargs):
        kwargs["loc"]        = "left"
        kwargs["ha"]         = kwargs.get("ha", "left")
        kwargs["va"]         = kwargs.get("va", "bottom")
        kwargs["fontsize"]   = kwargs.get("fontsize",   theme.title["size"])
        kwargs["fontdict"]   = kwargs.get("fontdict",   theme.title)
        kwargs["fontweight"] = kwargs.get("fontweight", theme.title["weight"])
        plt.title(text, **kwargs)
        return self 
    
    def r_title(self, text: str, **kwargs):
        kwargs["loc"]      = "right"
        kwargs["ha"]       = kwargs.get("ha", "right")
        kwargs["va"]       = kwargs.get("va", "bottom")
        kwargs["fontdict"] = kwargs.get("fontdict", theme.note)
        kwargs["color"]    = theme.accent
        plt.title(text, **kwargs)
        return self 
    
    def size(self, w, h):
        self.figure.set_size_inches(w, h)
        return self
    
    def legend(self, *args, **kwargs):
        kwargs["framealpha"]   = kwargs.get("framealpha",   theme.framealpha)
        kwargs["handlelength"] = kwargs.get("handlelength", theme.handlelength)
        plt.legend(*args, **kwargs)
        return self
    
    def format_xaxis(self, fmt = DATE_FMT):
        plt.gca().xaxis.set_major_formatter(DATE_FMT)
        plt.gca().xaxis.set_minor_formatter(DATE_FMT)
        return self 

    def save(self, filename: Path, **kwargs):
        if str(filename).endswith("tex"):
            tikzplotlib.save(filename, **kwargs)
            return self 
        kwargs["transparent"] = kwargs.get("transparent", str(filename).endswith("svg"))
        plt.savefig(filename, **kwargs)
        return self 

    def adjust(self, **kwargs):
        plt.subplots_adjust(**kwargs)
        return self 

    def show(self, **kwargs):
        plt.show(**kwargs)
        return self 

def plot_SIRD(model: NetworkedSIR, layout = (1,  4)) -> PlotDevice:
    """ plot all 4 available curves (S, I, R, D) for a given SIR  model """
    fig, axes = plt.subplots(layout[0], layout[1], sharex = True, sharey = True)
    t = list(range(len(model[0].Rt)))
    for (ax, model) in zip(axes.flat, model.units):
        s, = ax.semilogy(t, model.S, alpha=0.75, label="Susceptibles")
        i, = ax.semilogy(t, model.I, alpha=0.75, label="Infectious", )
        d, = ax.semilogy(t, model.D, alpha=0.75, label="Deaths",     )
        r, = ax.semilogy(t, model.R, alpha=0.75, label="Recovered",  )
        ax.label_outer()
    
    fig.legend([s, i, r, d], ["S", "I", "R", "D"], loc = "center right", borderaxespad = 0.1)
    return PlotDevice(fig)

def plot_curve(models: Sequence[NetworkedSIR], labels: Sequence[str], curve: str = "I"):
    """ plot specific epidemic curve """
    fig = plt.figure()
    for (model, label) in zip(models, labels):
        plt.semilogy(model.aggregate(curve), label = label, figure = fig)
    plt.legend() 
    plt.tight_layout()
    return PlotDevice(fig)

def gantt_chart(gantt_data, start_date: Optional[str] = None, show_cbar = True):
    """ create a Gantt chart showing adaptive control status (red/yellow/green) from set of simulations """
    gantt_df = pd.DataFrame(gantt_data, columns = ["district", "day", "beta", "R"])
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    if start_date:
        start_timestamp = pd.to_datetime(start_date)
        dates = [start_timestamp + datetime.timedelta(days = n) for n in gantt_df.day.unique()]
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

def predictions(date_range, model, color, bounds = [2.5, 97.5], curve = "dT"):
    mdn, min_, max_ = zip(*[np.percentile(_, [50] + bounds) for _ in model.__getattribute__(curve)])
    range_marker   = plt.fill_between(date_range, min_, max_, color = color, alpha = 0.3)
    median_marker, = plt.plot(date_range, mdn, color = color)
    return [(range_marker, median_marker), model.name]

def simulations(
    simulation_results: Sequence[Tuple[NetworkedSIR]], 
    labels: Sequence[str], 
    historical: Optional[pd.Series] = None, 
    historical_label: str = "Empirical Case Data", 
    curve: str = "dT", 
    smoothing: Optional[np.ndarray] = None, 
    semilog: bool = True) -> PlotDevice:
    """ plot simulation results for new daily cases and optionally show historical trends """

    aggregates = [tuple(model.aggregate(curve) for model in model_set) for model_set in simulation_results]

    policy_outcomes = list(zip(*aggregates))

    num_sims   = len(simulation_results)
    total_time = len(policy_outcomes[0][0])

    ranges: List[Dict[str, List]] = [{"max": [], "min": [], "mdn": [], "avg": []} for _ in range(len(policy_outcomes))]

    for (i, policy) in enumerate(policy_outcomes):
        for _ in range(total_time):
            curve_sorted = sorted([curve[_] for curve in policy])
            ranges[i]["min"].append(curve_sorted[0])
            ranges[i]["max"].append(curve_sorted[-1])
            ranges[i]["mdn"].append(curve_sorted[num_sims//2])
            ranges[i]["avg"].append(np.mean(curve_sorted))

    legends = []
    legend_labels = []
    if historical is not None:
        p, = plt.plot(historical.index, historical, 'k-', alpha = 0.8, zorder = 10)
        t  = [historical.index.max() + datetime.timedelta(days = n) for n in range(total_time)]
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
    plt.legend(legends, legend_labels, prop = dict(size = 20), handlelength = theme.handlelength, framealpha = theme.framealpha, loc = "best")

    plt.xlim(left = historical.index[0], right = t[-1])
    if semilog:
        plt.semilogy()
    set_tick_size(14)
    return PlotDevice()

def Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin = 0.5, ymax = 3, yaxis_colors = True, format_dates = True, critical_threshold = True, legend = True, legend_loc = "best"):
    """ plot Rt and associated confidence  intervals over time """
    CI_marker  = plt.fill_between(dates, Rt_CI_lower, Rt_CI_upper, color = BLK, alpha = 0.3)
    Rt_marker, = plt.plot(dates, Rt_pred, color = BLK, linewidth = 2, zorder = 5, solid_capstyle = "butt")
    if yaxis_colors: 
        plt.plot([dates[0], dates[0]], [2.5, ymax], color = RED, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
        plt.plot([dates[0], dates[0]], [1,    2.5], color = YLW, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
        plt.plot([dates[0], dates[0]], [ymin,   1], color = GRN, linewidth = 6, alpha = 0.9, solid_capstyle="butt", zorder = 10)
        plt.plot([dates[0], dates[0]], [ymin, ymax], color = "white", linewidth = 10, alpha = 1, solid_capstyle="butt", zorder = 9)
    if critical_threshold:
        plt.hlines(1, xmin=dates[0], xmax=dates[-1], zorder = 11, color = "black", linestyles = "dotted")
    plt.ylim(ymin, ymax)
    plt.xlim(left=dates[0], right=dates[-1])
    pd = PlotDevice()
    if legend:
        pd.legend_props = dict(framealpha = theme.framealpha, handlelength = theme.handlelength, loc = legend_loc)
        plt.legend([(CI_marker, Rt_marker)], [f"Estimated $R_t$ ({100*CI}% CI)"], **pd.legend_props)
    if format_dates:
        plt.gca().xaxis.set_major_formatter(DATE_FMT)
        plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    set_tick_size(theme.ticks["size"])
    pd.markers = {"Rt" : (CI_marker, Rt_marker)}
    return pd 

def daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, prediction_ts = None): 
    """ plots expected, smoothed cases from simulated annealing training """
    new_cases_dates = dates[-len(new_cases_ts):]
    exp_cases_dates = dates[-len(T_pred):]
    valid_idx   = [i for i in range(len(dates)) if dates[i] not in anomaly_dates] 
    T_CI_lower_rect = [min(l, u) for (l, u) in zip(T_CI_lower, T_CI_upper)]
    T_CI_upper_rect = [max(l, u) for (l, u) in zip(T_CI_lower, T_CI_upper)]
    observed_marker,   = plt.plot([d for d in new_cases_dates if d not in anomaly_dates], [new_cases_ts[i] for i in range(len(new_cases_ts)) if new_cases_dates[i] not in anomaly_dates], color = OBS_BLK, linewidth = 2, zorder = 8)
    anomalies_marker,  = plt.plot(anomaly_dates, anomalies, marker="o", mfc = "none", mec=ANOMALY_RED, lw = 0, zorder = 15)
    expected_marker,   = plt.plot(dates[-len(T_pred):], T_pred, color = CASE_BLU, marker=".", zorder = 10, lw = 0)
    expected_CI_marker = plt.fill_between(dates, T_CI_lower_rect, T_CI_upper_rect, label = f"", facecolor = CASE_BLU, alpha = 0.35)
    legends = [observed_marker, (expected_CI_marker, expected_marker)]
    labels  = ["observed cases (smoothed)", f"expected cases ({100*CI}% CI)"]
    (_, top) = plt.ylim()
    if prediction_ts:
        t_pred = [(dates[-1] + datetime.timedelta(days = i)) for i in range(len(prediction_ts[0][0]))]
        end = t_pred[-1]
        for (predictions, pred_CI_lower, pred_CI_upper, color, label) in prediction_ts:
            predicted_marker,   = plt.plot(t_pred, predictions, color = color, marker = ".", zorder = 9, lw = 0)
            predicted_CI_marker = plt.fill_between(t_pred, pred_CI_lower, pred_CI_upper, facecolor = color, alpha = 0.35)
            (_, top) = plt.ylim()
            plt.vlines(dates[-1], ymin = 0, ymax = top, color = "black", linestyles = "dotted")
            legends += [(predicted_CI_marker, predicted_marker)]
            labels  += [label]
    else: 
        end = dates[-1]
    plt.ylim(bottom = 0, top = top)
    legends += [anomalies_marker]
    labels  += ["anomalies"]
    plt.xlim(left = dates[0], right = end)
    plt.legend(legends, labels, prop = {'size': 14}, framealpha = theme.framealpha, handlelength = theme.handlelength, loc = "best")
    plt.gca().xaxis.set_major_formatter(DATE_FMT)
    plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    set_tick_size(14)
    return PlotDevice()

def choropleth(gdf, label_fn = lambda _: "", col = "Rt", title = "$R_t$", label_kwargs = {}, mappable = sm, fig = None, ax = None):
    """ display choropleth of locations by metric """
    gdf["pt"] = gdf["geometry"].centroid
    if not fig:
        fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left", fontdict = theme.label) 
    gdf.plot(color=[mappable.to_rgba(_) for _ in gdf[col]], ax = ax, edgecolors="black", linewidth=0.5, missing_kwds = {"color": theme.accent, "edgecolor": "white"})
    if label_fn is not None:
        for (_, row) in gdf.iterrows():
            label = label_fn(row)
            value = round(row[col], 2)
            ax.annotate(
                s = f"{label}{value}", 
                xy = list(row["pt"].coords)[0], 
                ha = "center", 
                fontfamily = theme.note["family"], 
                color = "black", **label_kwargs, 
                fontweight = "semibold",
                size = 12)\
                .set_path_effects([Stroke(linewidth = 2, foreground = "white"), Normal()])
    cbar_ax = fig.add_axes([0.90, 0.25, 0.01, 0.5])
    cb = fig.colorbar(mappable = mappable, orientation = "vertical", cax = cbar_ax)
    cbar_ax.set_title("$R_t$", fontdict = theme.note)
    
    return PlotDevice(fig)

def double_choropleth(gdf, label_fn = lambda _: "", Rt_col = "Rt", Rt_proj_col = "Rt_proj", titles = ["Current $R_t$", "Projected $R_t$ (1 Week)"], arrangement = (1, 2), label_kwargs = {}, mappable = sm):
    """ plot two choropleths side-by-side based on multiple metrics """
    gdf["pt"] = gdf["geometry"].centroid
    fig, (ax1, ax2) = plt.subplots(*arrangement)
    for (ax, title, col) in zip((ax1, ax2), titles, (Rt_col, Rt_proj_col)):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontdict = theme.label) 
        gdf.plot(color=[mappable.to_rgba(_) for _ in gdf[col]], ax = ax, edgecolors="black", linewidth=0.5, missing_kwds = {"color": theme.accent, "edgecolor": "white"})
    if label_fn is not None:
        for (_, row) in gdf.iterrows():
            label = label_fn(row)
            Rt_c, Rt_p = round(row[Rt_col], 2), round(row[Rt_proj_col], 2)
            a1 = ax1.annotate(s=f"{label}{Rt_c}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = theme.note["family"], color="white", **label_kwargs)
            a2 = ax2.annotate(s=f"{label}{Rt_p}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = theme.note["family"], color="white", **label_kwargs)
            a1.set_path_effects([Stroke(linewidth = 2, foreground = "black"), Normal()])
            a2.set_path_effects([Stroke(linewidth = 2, foreground = "black"), Normal()])
    cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    cb = fig.colorbar(mappable = mappable, orientation = "vertical", cax = cbar_ax)
    cbar_ax.set_title("$R_t$", fontdict = theme.note)
    
    return PlotDevice(fig)

def double_choropleth_v(*args, **kwargs):
    """ plot two choropleths (one on top of the other) based on multiple metrics """
    kwargs["arrangement"] = (2, 1)
    return double_choropleth(*args, **kwargs)

double_choropleth.horizontal = double_choropleth   # type: ignore
double_choropleth.vertical   = double_choropleth_v # type: ignore
