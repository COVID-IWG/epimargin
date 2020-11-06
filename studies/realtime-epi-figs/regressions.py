import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.utils import cwd
from scipy.optimize import minimize
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso,
                                  LinearRegression, LogisticRegression, Ridge)
from sklearn.preprocessing import minmax_scale
from sklearn.svm import LinearSVC

data = cwd()/"example_data"
df = pd.read_csv(data/"metro_state_policy_evaluation.csv").dropna()
df["Rt_binarized"] = (df["RR_pred"] >= 1).astype(int)
X = pd.concat([
    df.drop(columns = 
        [col for col in df.columns if col.startswith("RR_")]    + 
        [col for col in df.columns if col.startswith("metro_")] +
        ["metro-state", "date", "state", "state_name", "start_stay_at_home", 
        "end_stay_at_home", "mask_mandate_all", "metro_outbreak_start", "threshold_ind", "cbsa_fips", 
        "new_cases_ts", "daily_confirmed_cases"]),
    # pd.get_dummies(df.state_name, prefix = "state_name")
], axis = 1)


# set up metro fixed effects
metros = [col for col in df.columns ]