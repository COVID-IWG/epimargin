from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from etl import *
from adaptive.utils import cwd, days


if __name__ == '__main__':

    root = cwd()
    data = root/'data'
    figs = root/'figs'

    # load meta data for metro-state aggregations & filter to top 100 metro areas
    county_populations = load_us_county_data('covid_county_population_usafacts.csv')
    county_populations = county_populations[county_populations['county_name'] != 'Statewide Unallocated']
    metro_areas = load_metro_areas(data/'county_metro_state_walk.csv').rename(columns={'state_codes': 'state', 'county_fips': 'countyfips'})
    top_metros = get_top_metros(county_populations, metro_areas)

    top_metros = county_populations.merge(top_metros[['countyfips','cbsa_fips']], on='countyfips')
    top_metros = top_metros.groupby(['cbsa_fips', 'state']).apply(pop_prop_col)
    top_metros['metro-state'] = top_metros['cbsa_fips'].astype(int).astype(str) + '_' + top_metros['state']

    # load county level daily case data 
    case_df = load_us_county_data('covid_confirmed_usafacts.csv')
    county_case_ts = get_case_timeseries(case_df).merge(top_metros, on='countyfips', how='inner')

    # load county level google mobility data, and impute/drop missing data
    ## N.B parks_percent_change_from_baseline and transit_stations_percent_change_from_baseline still have a lot of missing --> drop these columns from analysis
    us_mobility = load_country_google_mobility('US').rename(columns={'sub_region_1': 'state_name', 'census_fips_code': 'countyfips'})
    county_mobility_ts = us_mobility[~(us_mobility['countyfips'].isna())].set_index(['countyfips', 'date'])
    county_mobility_ts = impute_missing_mobility(county_mobility_ts)
    county_mobility_imputed, lst_cnt = remain_county(county_mobility_ts.merge(county_populations[['countyfips','population']], on='countyfips'))

    # metro-state level aggregation
    metro_state_cases = pd.DataFrame(county_case_ts.groupby(['metro-state', 'date'])['daily_confirmed_cases'].sum())
    metro_state_mobility = metro_state_mobility_agg(county_mobility_imputed.merge(top_metros, on='countyfips', how='inner'))

    # load rt daily data
    rt_drop_cols = ['RR_pred_rtliveold', 'RR_CI_lower_rtliveold', 'RR_CI_upper_rtliveold']
    metro_state_rt = pd.read_csv(data/'+rt_estimates_comparison.csv', parse_dates=['date']).iloc[:,1:].rename(columns={'cbsa_fips_state':'metro-state'}).set_index(['metro-state','date'])
    metro_state_rt.drop(columns=rt_drop_cols, inplace=True)

    # create metro-state aggregated df
    metro_state_df = metro_state_mobility.join(metro_state_cases.join(metro_state_rt)).join(top_metros[['metro-state','state', 'cbsa_fips']].drop_duplicates().set_index('metro-state'))

    # add state level intervention dummies
    state_interventions = state_level_intervention_data(data/'COVID-19 US state policy database_08_03_2020.xlsx').reset_index().rename(columns={'STATE':'state_name'})
    metro_state_df = metro_state_df.reset_index().merge(state_interventions, on='state').set_index(['metro-state', 'date'])
    metro_state_df = metro_state_df.groupby(['metro-state']).apply(fill_dummies, 'stay_at_home', 'start_stay_at_home', 'end_stay_at_home')
    metro_state_df = metro_state_df.groupby(['metro-state']).apply(fill_dummies, 'mask_mandate', 'mask_mandate_all')

    # add dummy to only include places once their outbreak has started - seems that 10 cases is the threshold used 
    metro_state_df = start_outbreak_dummy(metro_state_df)

    # add dummies for metro areas
    metro_state_df = get_metro_dummies(metro_state_df)

    metro_state_df.to_csv(data/'metro_state_policy_evaluation.csv')
