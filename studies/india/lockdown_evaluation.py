from pathlib import Path

import numpy as np

import etl
from adaptive.model import Model, ModelUnit
from adaptive.utils import *

root = "."


df = case_data(Path(root), schemas.india_v2)
ts = timeseries(df)
gr = growth_rates(ts)

def units():
    pass 

migrations = []
lockeddown = np.zeros()

RR0_voluntary = {}
RR0_mandatory = {}

seed = 11235813

# policy A: 03 may release
release_03_may = lambda: Model(units(), random_seed = seed)\
    .set_parameters(RR0 = RR0_mandatory)\
    .run(10*days,  migrations = lockeddown)\
    .set_parameters(RR0 = RR0_voluntary)\
    .run(180*days, migrations = migrations)

# policy B: 31  May release
release_31_may = lambda: Model(units(), random_seed = seed)\
    .set_parameters(RR0 = RR0_mandatory)\
    .run(10*days + 4*weeks,  migrations = lockeddown)\
    .set_parameters(RR0 = RR0_voluntary)\
    .run(152*days, migrations = migrations)

# policy C: adaptive release

if __name__ == "__main__":
    root = cwd()