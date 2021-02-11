from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from studies.age_structure.commons import *
from sklearn.preprocessing import MinMaxScaler
from linearmodels import PanelOLS

# dcons = average daily consumption per household
# rc    = percent change in daily consumption per household relative to 2019m6
df = pd.read_stata("data/datareg.dta")
# df["dcons_scaled"] = MinMaxScaler().fit_transform(df[["rc"]]) 

rc = df[["districtnum", "month_code", "rc"]].set_index(["districtnum", "month_code"])

exog_cols = ["I_cat", "D_cat", "I_cat_national", "D_cat_national"]
for col in exog_cols:
    df[col] = pd.Categorical(df[col])

PanelOLS(rc, df[exog_cols])

# # pandas needs explicit dummy columns 
# X = pd.concat([pd.get_dummies(df[column], prefix = prefix) for (column, prefix) in {
#     "month_code"    : "month",
#     "districtnum"   : "district",
#     "I_cat"         : "I_cat_loc",
#     "D_cat"         : "D_cat_loc",
#     "I_cat_national": "I_cat_nat",
#     "D_cat_national": "D_cat_nat"
# }.items()], axis = 1)

# reg = sm.OLS(df["dcons"], X).fit()

# PanelOLS(df[["districtnum", "month_code", "rc"]].set_index(["districtnum", "month_code"]), X)