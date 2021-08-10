import matplotlib.pyplot as plt
import numpy as np
from epimargin.models import NetworkedSIR, SIR
from epimargin.plots import plot_SIRD
from scipy.spatial import distance_matrix

names  = [f"state{i}" for i in ( 1, 2, 3, 4)]
pops   = [k * 1000    for k in (10, 1, 2, 2)]

# inverse distance transmission matrix
centroids = [(0, 0), (1, 0), (0, 1), (2, 0)]
P = distance_matrix(centroids, centroids)
P[P != 0] = P[P != 0] ** -1.0 # we tried inverse square but Zünd et al. suggest inverse
P *= np.array(pops)[:, None]  # weight by destination population
P /= P.sum(axis = 0)          # normalize 

# states 
units = lambda Rt0: [SIR(name = name, population = pop, Rt0 = Rt0) for (name, pop) in zip(names, pops)]

# run and plot 
networked_model = NetworkedSIR(units(1.9), P, 0).run(200)
plot_SIRD(networked_model)\
    .title("Four-State Toy Model")\
    .xlabel("Time")\
    .ylabel("S/I/R/D")\
    .show()
