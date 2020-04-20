import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix

from model import Model, ModelUnit

mlp.rcParams['font.sans-serif'] = "PT Sans Regular"
mlp.rcParams['font.family'] = "sans-serif"
sns.set_style("whitegrid")
sns.despine()

# setting up plotting 
def plot(model: Model, filename: Optional[str] = None) -> mlp.figure.Figure:
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

# inverse square law transmission matrix
centroids = [(0, 0), (1, 0), (0, 1), (2, 0)]
P = distance_matrix(centroids, centroids)
P[P != 0] = P[P != 0] ** -2.0
P /= P.sum(axis = 1)

# states 
units = lambda rr0 = 1.9: [
    ModelUnit(name = "state1", population = 10_000, RR0 = rr0),
    ModelUnit(name = "state2", population =  1_000, RR0 = rr0),
    ModelUnit(name = "state3", population =  2_000, RR0 = rr0),
    ModelUnit(name = "state4", population =  2_000, RR0 = rr0)
]

# run and plot 
m1_9 = Model(200, units(1.9), P).run()
plot(m1_9, "example_model_1.9.png")

m0_9 = Model(200, units(0.9), P).run()
plot(m0_9, "example_model_0.9.png")
