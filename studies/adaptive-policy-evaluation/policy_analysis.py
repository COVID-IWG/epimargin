from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib.lines import Line2D
from adaptive.utils import cwd, days

def load_dataframe(data_path: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates = ['date'])
    return df[df['date'] >= pd.to_datetime('2020-03-01')]

def create_policy_wave_dummies(county_policy_df: pd.DataFrame, policy_col: str) -> pd.DataFrame:
    grouped = pd.DataFrame(county_policy_df.groupby(['cbsa_fips','metro-state']).apply(lambda x: x['date'][x[policy_col] == 1].min()))
    grouped['rank'] = grouped.groupby("cbsa_fips")[0].rank(method = 'dense', ascending = True)
    wave_dummies = pd.get_dummies(grouped['rank'])
    wave_dummies.columns = [policy_col + '_wave_' + str(x) for x in wave_dummies.columns]
    return wave_dummies.reset_index()

def regression(county_policy_df: pd.DataFrame):
    # drop very small number of missing
    # county_policy_df.dropna(inplace=True)
    predictor_cols = [x for x in county_policy_df.columns if x.startswith('time') or x.startswith('metro') or x.endswith('baseline') or x.startswith('intervention')]
    X = county_policy_df[predictor_cols].values
    X = sm.add_constant(X, has_constant='add')
    y = county_policy_df['daily_confirmed_cases_lag_-7'].values
    res = sm.OLS(y, X).fit()
    print(res.summary())
    return predictor_cols

if __name__ == '__main__':
    root = cwd()
    data = root/"data"

    intervention = 'stay_at_home'

    county_policy_df = load_dataframe(data/'metro_state_policy_evaluation.csv')

    # drop google cols with lots of missing
    county_policy_df.drop(columns=['parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline'], inplace=True)

    # drop most recent 2 weeks of dataframe
    date = dt.datetime.today() - pd.Timedelta(14, unit='d')
    county_policy_df = county_policy_df[county_policy_df['date'] <= date]

    policy_wave_dummies = create_policy_wave_dummies(county_policy_df, intervention).set_index(['metro-state'])
    county_policy_df = county_policy_df.set_index(['metro-state']).join(policy_wave_dummies.iloc[:, 1:]).reset_index()
    county_policy_df[policy_wave_dummies.columns].fillna(0, inplace=True)
    county_policy_df.rename(columns={intervention: 'time_dummy_' + intervention}, inplace=True)

    for wave_col in [x for x in county_policy_df.columns if x.startswith(intervention)]:
        county_policy_df['time_dummy*' + wave_col] = county_policy_df['time_dummy_' + intervention] * county_policy_df[wave_col]

    county_policy_df.drop(columns=['metro_outbreak_start'], inplace=True)
    county_policy_df.set_index(['metro-state','date'], inplace=True)

    regression(county_policy_df)

