# "mission begin again" policies 

from itertools import product
from typing import Dict, Optional

import numpy as np

from adaptive.model import Model
from adaptive.utils import days, weeks

ORANGE_ZONES = ["PUNE", "SOLAPUR", "AURANGABAD", "NASHIK", "DHULE", "JALGAON", "AKOLA", "AMARAVATI", "NAGPUR"]

def simulate_MBA(
    model: Model, 
    initial_run: int, # time under strict lockdown 
    phased_time: int, # time under MBA
    total_time: int,  # total time to run model 
    lockdown: np.matrix, 
    migrations: np.matrix, 
    R_m: Dict[str, float],
    beta_v: Dict[str, float], 
    beta_m: Dict[str, float], 
    adjacency: Optional[np.matrix] = None) -> Model:
    n = len(model)
    
    # run strict lockdown 
    model.set_parameters(RR0 = R_m)\
         .run(initial_run, lockdown)
    
    for district in ORANGE_ZONES:
        beta = beta_v[district] - ((beta_v[district] - beta_m[district])/2.0)
        model[district].beta[-1] = beta
        model[district].RR0 = beta * model[district].gamma
    
    phased_migration = migrations.copy()
    for (i, j) in product(range(n), range(n)):
        if model[i].name not in ORANGE_ZONES or model[j].name not in ORANGE_ZONES:
            phased_migration[i, j] = 0
    model.run(phased_time, phased_migration)

    model.run(total_time - phased_time - initial_run, migrations)
    
    return model 
