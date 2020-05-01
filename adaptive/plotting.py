from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .model import Model

sns.set(style = "whitegrid", font = "Fira Code")
sns.despine()

# setting up plotting 
def plot_SIRD(model: Model, title, xlabel, ylabel, subtitle, filename: Optional[Path] = None) -> mlp.figure.Figure:
    fig, axes = plt.subplots(1, 4, sharex = True, sharey = True)
    fig.suptitle(title)
    t = list(range(len(model[0].RR)))
    for (ax, model) in zip(axes.flat, model.units):
        s = ax.semilogy(t, model.S, alpha=0.75, label="Susceptibles")
        i = ax.semilogy(t, model.I, alpha=0.75, label="Infectious", )
        d = ax.semilogy(t, model.D, alpha=0.75, label="Deaths",     )
        r = ax.semilogy(t, model.R, alpha=0.75, label="Recovered",  )
        ax.set(title = subtitle, xlabel = xlabel, ylabel = ylabel)
        ax.label_outer()
    
    fig.legend([s, i, r, d], labels = ["S", "I", "R", "D"], loc="center right", borderaxespad=0.1)
    plt.subplots_adjust(right=0.85)
    if filename: 
        plt.savefig(filename)
    return fig

def plot_curve(models: Sequence[Model], labels: Sequence[str], title, xlabel, ylabel, subtitle = None, curve: str = "I", filename = None):
    fig = plt.figure()
    for (model, label) in zip(models, labels):
        plt.semilogy(model.aggregate(curve), label = label, figure = fig)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if subtitle:
        plt.title(subtitle)
    plt.legend() 
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi = 600)
    return fig

def gantt_chart(gantt_data, xlabel, title, filename: Optional[Path] = None):
    gantt_df = pd.DataFrame(gantt_data, columns = ["district", "day", "beta", "R"])
    gantt_pv = gantt_df.pivot("district", "day", values = ["beta", "R"])
    ax = sns.heatmap(gantt_pv["beta"], linewidths = 2, alpha = 0.8, 
        annot = gantt_pv["R"], annot_kws={"size": 8},
        cmap = ["#38AE66", "#FFF3B4", "#FD8B5A", "#D63231"],
        cbar = True,
        yticklabels = gantt_df["district"].unique(),
        cbar_kws = {
            "ticks":[0.5, 1, 2, 2.5], 
            "label": "Mobility", 
            "format": mlp.ticker.FuncFormatter(lambda x, pos: {0.5: "voluntary", 1: "cautionary", 2: "partial", 2.5: "restricted"}[x]), 
            "orientation": "horizontal", 
            "aspect": 50, 
            "drawedges": True,
            "fraction": 0.05,
            "pad": 0.10, 
            "shrink": 0.5
        }
    )
    ax.set(xlabel = xlabel, ylabel = None)
    plt.suptitle(title)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.10, bottom=0.10)
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")