from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from studies.age_structure.commons import *
from linearmodels import PanelOLS

# dcons = average daily consumption per household
# rc    = percent change in daily consumption per household relative to 2019m6
df = pd.read_stata("data/datareg.dta")

index = ["districtnum", "month_code"]
rc = df[index + ["rc"]].set_index(index)

exog_cols = ["I_cat", "D_cat", "I_cat_national", "D_cat_national"]
for col in exog_cols:
    df[col] = pd.Categorical(df[col])

exog = df[index + exog_cols].set_index(index)

PanelOLS(rc, exog, entity_effects = True)
