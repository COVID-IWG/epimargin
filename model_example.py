from scipy.spatial import distance_matrix

from model import Model, ModelUnit

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
Model(200, units(1.9), P)\
    .run()\
    .show("example_model_1.9.png")

Model(200, units(0.9), P)\
    .run()\
    .show("example_model_0.9.png")
