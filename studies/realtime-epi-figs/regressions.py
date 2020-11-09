import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.utils import cwd

import statsmodels.api as sm


# load data 
data = cwd()/"example_data"
df = pd.read_csv(data/"metro_state_policy_evaluation.csv").dropna()
df["Rt"] = df["RR_pred"]
df["Rt_binarized"] = (df["RR_pred"] >= 1).astype(int)

drop_cols = [col for col in df.columns if col.startswith("RR_")] + [
    "metro-state", "date", "state", "state_name", "start_stay_at_home", "cbsa_fips",
    "end_stay_at_home", "mask_mandate_all", "metro_outbreak_start", "threshold_ind",
    "new_cases_ts", "daily_confirmed_cases", "religious_exception_stay_home"
]
df.drop(columns = drop_cols, inplace = True)
df.rename(lambda x: x.replace(".0", ""), axis = 1, inplace=True) 

# set up metro fixed effects (some metro cols are all 0)
metros = [col for col in df.columns if col.startswith("metro") if df[col].sum() > 0]
# drop one metro to avoid multicollinearity
reference_metro, *metros = metros
drop_metro_cols = [col for col in df.columns if col.startswith("metro_") and col.endswith("0") and col not in metros]
df.drop(columns = drop_metro_cols, inplace = True)

# parks mobility is wonky, clip at central 95% of distribution
quantiles = df.parks_percent_change_from_baseline.quantile([0.025, 0.975])
df = df[df.parks_percent_change_from_baseline.between(*quantiles)]
base_weight = 0.5

# run regressions
# 1: which policies reduce Rt?
covariates = [col for col in df.columns if not col.startswith("Rt")]
penalty_weights = [base_weight] + [0 if col.startswith("metro") else base_weight for col in covariates]
continuous_model = sm.OLS.from_formula("Rt ~ " + " + ".join(covariates), data = df)
continuous_model.fit()
cov_params = continuous_model.normalized_cov_params
continuous_results = continuous_model.fit_regularized(alpha = penalty_weights, L1_wt = 0.9)
approx_summary = sm.regression.linear_model.OLSResults(continuous_model, continuous_results.params, cov_params).summary()
print(approx_summary)
print(approx_summary.as_latex())

df_orig = df.copy()

# 2: which policies bring Rt < 1?
cuts = [-100, -50, 0, 25, 50, 75, 100]
for k in [col for col in df.columns if col.endswith("_baseline")]:
    mobilty_type = k.replace("_percent_change_from_baseline", "")
    df = pd.concat([df, 
        pd.get_dummies(pd.cut(df[k], cuts))\
            .rename(lambda x: "L_" + mobilty_type + str(x).replace("(", "").replace("]", "").replace(" ", "").replace("-", "n").replace(",", "_"), axis = 1)
    ], axis = 1)

df.drop(columns = [col for col in df.columns if col.endswith("_baseline")], inplace=True)

base_weight = 1
covariates = [col for col in df.columns if not col.startswith("Rt")]
penalty_weights = [base_weight] + [0 if col.startswith("metro") else base_weight for col in covariates]
binarized_model = sm.Probit.from_formula("Rt_binarized ~ " + " + ".join(covariates), data = df)
binarized_results = binarized_model.fit_regularized(alpha = penalty_weights, L1_wt = 0.9)
print(binarized_results.summary())

non_fe = binarized_results.params[~binarized_results.params.index.str.startswith("metro")]
non_zero = non_fe[non_fe != 0].round(2)
print(non_zero.sort_values())