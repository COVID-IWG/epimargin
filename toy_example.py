import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 

from adaptive.model import Model, ModelUnit
from adaptive.plotting import plot_SIRD

names  = [f"state{i}" for i in ( 1, 2, 3, 4)]
pops   = [k * 1000    for k in (10, 1, 2, 2)]

# inverse distance transmission matrix
centroids = [(0, 0), (1, 0), (0, 1), (2, 0)]
P = distance_matrix(centroids, centroids)
P[P != 0] = P[P != 0] ** -1.0 # we tried inverse square but Zünd et al. suggest inverse
P *= np.array(pops)[:, None]  # weight by destination population
P /= P.sum(axis = 0)          # normalize 

# states 
units = lambda rr0: [ModelUnit(name = name, population = pop, RR0 = rr0) for (name, pop) in zip(names, pops)]

# run and plot 
m1_9 = Model(units(1.9), P, 0).run(200)
plot_SIRD(m1_9, title = "Four-State Toy Model", xlabel="Time", ylabel="S/I/R/D", subtitle="No Adaptive Controls")
plt.show()