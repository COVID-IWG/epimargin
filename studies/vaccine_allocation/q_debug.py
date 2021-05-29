import dask
import numpy as np
import pandas as pd
from epimargin.models import Age_SIRVD
from epimargin.utils import annually, normalize, percent, years
from studies.vaccine_allocation.commons import *
from studies.vaccine_allocation.epi_simulations import districts_to_run, MORTALITY, prioritize

import warnings
warnings.filterwarnings("error")

num_sims = 10
phi = 1000 * percent * annually

(
    state_code, 
    sero_0, N_0, sero_1, N_1, sero_2, N_2, sero_3, N_3, sero_4, N_4, sero_5, N_5, sero_6, N_6, N_tot, 
    Rt, Rt_upper, Rt_lower, S0, I0, R0, D0, dT0, dD0, V0, T_ratio
) = districts_to_run.iloc[0]

Sj0 = np.array([(1 - sj) * Nj for (sj, Nj) in zip([sero_0, sero_1, sero_2, sero_3, sero_4, sero_5, sero_6], [N_0, N_1, N_2, N_3, N_4, N_5, N_6])])
Sj0 = prioritize(V0, Sj0.copy()[None, :], MORTALITY)[0]

model = Age_SIRVD(
    name        = "q0_debug", 
    population  = N_tot - D0, 
    dT0         = (np.ones(num_sims) * dT0).astype(int), 
    Rt0         = 0 if S0 == 0 else Rt * N_tot / S0,
    S0          = np.tile( Sj0,        num_sims).reshape((num_sims, -1)),
    I0          = np.tile((fI * I0).T, num_sims).reshape((num_sims, -1)),
    R0          = np.tile((fR * R0).T, num_sims).reshape((num_sims, -1)),
    D0          = np.tile((fD * D0).T, num_sims).reshape((num_sims, -1)),
    mortality   = np.array(list(OD_IFRs.values())),
    infectious_period = infectious_period,
    random_seed = 0,
)
model.dD_total[0] = np.ones(num_sims) * dD0
model.dT_total[0] = np.ones(num_sims) * dT0

num_doses = phi * (S0 + I0 + R0)
for t in range(365):
    if t <= 1/phi:
        dV_mortality = prioritize(num_doses, model.N[-1], MORTALITY).clip(0) 
    else: 
        dV_mortality = np.zeros((num_sims, 7))
    model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)