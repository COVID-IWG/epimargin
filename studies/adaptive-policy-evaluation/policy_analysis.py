from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib.lines import Line2D
from etl import load_us_county_data, load_country_google_mobility, load_intervention_data, get_case_timeseries, add_lag_cols, load_metro_areas, load_rt_estimations, fill_dummies, load_county_mask_data, add_mask_dummies
from adaptive.utils import cwd, days

# def regression_one():
#     # results = smf.ols('daily_confirmed_cases ~ retail_and_recreation_percent_change_from_baseline + transit_stations_percent_change_from_baseline + \
#     #                 workplaces_percent_change_from_baseline + residential_percent_change_from_baseline', data=full_df.loc['Illinois','Cook County']).fit()
#     # print(results.summary())

#     X = cook_county.iloc[:,2:].values

#     X = sm.add_constant(X)
#     X = sm.add_constant(X, has_constant='add')
#     y = cook_county['daily_confirmed_cases'].values
#     res = sm.OLS(y, X).fit()
#     pass

def load_county_dataset(data_path: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=['date'])
    return df[]

def create_policy_wave_dummies(county_df: pd.DataFrame, policy_col: str) -> pd.DataFrame:
    grouped = pd.DataFrame(county_df.groupby(['cbsa_fips','county_name']).apply(lambda x: x['date'][x[policy_col] == 1].min()))
    grouped['rank'] = grouped.groupby("cbsa_fips")[0].rank(method = 'dense', ascending = False)
    wave_dummies = pd.get_dummies(grouped['rank'])
    wave_dummies.columns = ['wave_' + str(x) for x in wave_dummies.columns]
    return wave_dummies.reset_index()


create_policy_wave_dummies(df, 'intervention_public_schools')