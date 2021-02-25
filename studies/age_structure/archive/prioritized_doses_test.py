import numpy as np
from adaptive.models import SIR
from adaptive.policy import PrioritizedAssignment
from studies.age_structure.commons import *

vp = PrioritizedAssignment(
    daily_doses    = 100,
    effectiveness  = 1, 
    S_bins         = np.array([
        [10, 20, 30, 40, 50, 50, 60],
        [10, 20, 30, 40, 50, 50, 45],
        [10, 20, 30, 40, 50, 50, 0]
    ]),
    I_bins         = np.array([
        [0, 0, 0, 5, 6, 7, 10],
        [0, 0, 0, 5, 6, 7, 45],
        [0, 0, 0, 5, 6, 7, 70]
    ]), 
    age_ratios     = np.array([0.2, 0.2, 0.25, 0.1, 0.1, 0.1, 0.05]),
    IFRs           = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04]), 
    prioritization = [6, 5, 4, 3, 2, 1],
    label          = "testpolicy"
)

result = np.array([
    [10, 20, 30, 40, 50, 10, 0],
    [10, 20, 30, 40, 45,  0, 0],
    [10, 20, 30, 40, 0,   0, 0],

])

model = SIR("testmodel", 100000)

p = vp.S_bins.copy()

p0 = np.array([60, 50, 50, 40, 30, 20, 10])
d = 100
for i in range(len(p0)-1):
    if p0[i] <= d:
        d -= p0[i]
        p0[i] = 0
    else:
        p0[i] -= d
    print("post", i, d, p0)
p0


d = 10
p = vp.S_bins[:, ::-1].copy()

def distribute(p, doses):
    q = np.where(p.cumsum(axis = 1) <= doses[:, None], 0, p)
    q[np.arange(len(q)), (q != 0).argmax(axis = 1)] -= (doses - np.where(p.cumsum(axis = 1) > doses[:, None], 0, p).sum(axis = 1))
    return q, p-q

# def distribute(p0, doses):
#     p = p0.copy()
#     p[np.arange(len(p)), (p != 0).argmax(axis = 1)] -= doses - np.where(p.cumsum(axis = 1) >= doses, 0, p).sum(axis = 1) # set elements 
#     return p, p0 - p