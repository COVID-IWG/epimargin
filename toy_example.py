from typing import Optional

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import distance_matrix

from model import Model, ModelUnit

mlp.rcParams['font.sans-serif'] = "PT Sans Regular"
mlp.rcParams['font.family'] = "sans-serif"
sns.set_style("whitegrid")
sns.despine()

# setting up plotting 
def plot_SIRD(model: Model, filename: Optional[str] = None) -> mlp.figure.Figure:
    fig, axes = plt.subplots(1, 4, sharex = True, sharey = True)
    fig.suptitle("Four-State Toy Example (No Adaptive Controls; $R_0^{(0)} = " + str(model.units[0].RR0) + "$)")
    t = list(range(model.num_days + 1))
    for (ax, model) in zip(axes.flat, model.units):
        s = ax.semilogy(t, model.S, alpha=0.75, label="Susceptibles")
        i = ax.semilogy(t, model.I, alpha=0.75, label="Infectious", )
        d = ax.semilogy(t, model.D, alpha=0.75, label="Deaths",     )
        r = ax.semilogy(t, model.R, alpha=0.75, label="Recovered",  )
        ax.set(xlabel = "# days", ylabel = "S/I/R/D", title = f"{model.name} (initial pop: {model.pop0})")
        ax.label_outer()
    
    fig.legend([s, i, r, d], labels = ["S", "I", "R", "D"], loc="center right", borderaxespad=0.1)
    plt.subplots_adjust(right=0.85)
    if filename: 
        plt.savefig(filename)
    return fig

names = [f"state{i}" for i in ( 1, 2, 3, 4)]
pops  = [k * 1000    for k in (10, 1, 2, 2)]
states = zip(names, pops)

# inverse distance transmission matrix
centroids = [(0, 0), (1, 0), (0, 1), (2, 0)]
P = distance_matrix(centroids, centroids)
P[P != 0] = P[P != 0] ** -1.0 # we tried inverse square but Zünd et al. suggest inverse
P *= np.array(pops)[:, None]  # weight by destination population
P /= P.sum(axis = 0)          # normalize 

# states 
units = lambda rr0 = 1.9: [ModelUnit(name = name, population = pop, RR0 = rr0) for (name, pop) in states]


# run and plot 
m1_9 = Model(200, units(1.9), P).run()
plot_SIRD(m1_9)
plt.show()
