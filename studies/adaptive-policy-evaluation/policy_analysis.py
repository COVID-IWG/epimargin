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

def load_county_dataset(data_path: pd.DataFrame, intervention_cols: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates = ['date'])
    df = df[df['threshold_ind'] == 1].iloc[:, :-9]
    drop_cols = [x for x in df.columns if x.startswith('intervention') and x not in intervention_cols]
    return df.drop(columns=drop_cols)

def add_metro_area_dummies(county_policy_df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(county_policy_df['cbsa_fips'])
    dummies.columns = ['metro_' + str(x) for x in dummies.columns]
    return county_policy_df.join(dummies)

def create_policy_wave_dummies(county_policy_df: pd.DataFrame, policy_col: str) -> pd.DataFrame:
    grouped = pd.DataFrame(county_policy_df.groupby(['cbsa_fips','countyfips']).apply(lambda x: x['date'][x[policy_col] == 1].min()))
    grouped['rank'] = grouped.groupby("cbsa_fips")[0].rank(method = 'dense', ascending = True)
    grouped = grouped[grouped['rank'] <= 2]
    wave_dummies = pd.get_dummies(grouped['rank'])
    wave_dummies.columns = [policy_col + '_wave_' + str(x) for x in wave_dummies.columns]
    return wave_dummies.reset_index()

def regression(county_policy_df: pd.DataFrame):
    county_policy_df.dropna(inplace=True)
    predictor_cols = [x for x in county_policy_df.columns if x.startswith('time') or x.startswith('metro') or x.endswith('baseline') or x.startswith('intervention')]
    X = county_policy_df[predictor_cols].values
    X = sm.add_constant(X, has_constant='add')
    y = county_policy_df['daily_confirmed_cases'].values
    res = sm.OLS(y, X).fit()
    print(res.summary())
    return predictor_cols

if __name__ == '__main__':
    root = cwd()
    data = root/"data"

    interventions = ['intervention_stay_at_home']

    county_policy_df = load_county_dataset(data/'county_level_policy_evaluation.csv', interventions)
    county_policy_df = add_metro_area_dummies(county_policy_df)

    # drop google cols with lots of missing
    county_policy_df.drop(columns=['parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline'], inplace=True)

    # drop most recent 2 weeks of dataframe
    date = dt.datetime.today() - pd.Timedelta(14, unit='d')
    county_policy_df = county_policy_df[county_policy_df['date'] <= date]

    for intervention in interventions:
        policy_wave_dummies = create_policy_wave_dummies(county_policy_df, intervention).set_index(['cbsa_fips','countyfips'])
        county_policy_df = county_policy_df.set_index(['cbsa_fips','countyfips']).join(policy_wave_dummies).reset_index()
        county_policy_df[policy_wave_dummies.columns].fillna(0, inplace=True)
        county_policy_df.rename(columns={intervention: 'time_dummy_' + intervention}, inplace=True)

        for wave_col in [x for x in county_policy_df.columns if x.startswith(intervention)]:
            county_policy_df['time_dummy*' + wave_col] = county_policy_df['time_dummy_' + intervention] * county_policy_df[wave_col]

    if len(interventions) > 1:
        for i in range(len(interventions)-1):
            county_policy_df[interventions[i] + '*' + interventions[i+1]] = county_policy_df[interventions[i]] * county_policy_df[interventions[i+1]]

    county_policy_df.drop(columns=['metro_outbreak_start'], inplace=True)

    regression(county_policy_df)

