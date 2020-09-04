from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from etl import *
from adaptive.utils import cwd, days


if __name__ == "__main__":

    root = cwd()
    data = root/"data"
    figs = root/"figs"

    case_df = load_us_county_data("covid_confirmed_usafacts.csv")
    case_timeseries = get_case_timeseries(case_df)
    
    us_mobility = load_country_google_mobility("US").rename(columns={"sub_region_1": "state_name", "census_fips_code": "countyfips"})

    county_populations = load_us_county_data("covid_county_population_usafacts.csv")
    county_populations = county_populations[county_populations['county_name'] != 'Statewide Unallocated']

    # county level
    county_case_ts = pd.DataFrame(case_timeseries[case_timeseries.index.get_level_values(level=1) != 0])
    county_mobility_ts = us_mobility[~(us_mobility["countyfips"].isna())].set_index(["state_name", "countyfips", "date"])

    county_df = county_case_ts.join(county_mobility_ts).join(county_populations.set_index(['state_name','countyfips'])).reset_index().set_index(["state_name", "countyfips", "date"])        

    # metro-state aggregation
    metro_areas = load_metro_areas(data/"county_metro_state_walk.csv").rename(columns={"state_codes": "state", "county_fips": "countyfips"})

    county_case_ts = county_case_ts.join(metro_areas.set_index(["state_name", "countyfips"])["cbsa_fips"])
    metro_state_cases = pd.DataFrame(county_case_ts.groupby(['cbsa_fips','state_name','date'])['daily_confirmed_cases'].sum())

    county_mobility_ts = county_mobility_ts.join(metro_areas.set_index(["state_name", "countyfips"]))
    metro_state_mobility = county_mobility_ts.groupby(['cbsa_fips','state_name','date']).mean() # need to make this weighted by population

    metro_state_df = metro_state_cases.join(metro_state_mobility, how='inner').sort_index()

    state_interventions = state_level_intervention_data(data/"COVID-19 US state policy database_08_03_2020.xlsx").reset_index().rename(columns={'STATE':'state_name'})
    metro_state_df = metro_state_df.join(state_interventions.set_index('state_name'))

    metro_state_df = metro_state_df.groupby(['cbsa_fips','state_name']).apply(fill_dummies, 'stay_at_home', 'start_stay_at_home', 'end_stay_at_home')
    metro_state_df = metro_state_df.groupby(["cbsa_fips","state_name"]).apply(add_lag_cols, ["daily_confirmed_cases"])
    
    # add dummy to only include places once their outbreak has started - seems that 10 cases is the threshold used 
    metro_state_df = start_outbreak_dummy(metro_state_df)

    # filter to top 100 metro areas
    # county_df_top_metros = filter_top_metros(county_df)


    # county_interventions = interventions[~(interventions.index.get_level_values(0) == interventions.index.get_level_values(1))]
    # county_mask_policy = load_county_mask_data(data/"county_mask_policy.csv")

    # political affiliation
    # vote_df = poli_aff(data/"countypres_2000_2016.csv").rename(columns={'state':'state_name'}).set_index(['state_name', 'countyfips'])
    # county_df_top_metros = county_df_top_metros.join(vote_df)

    # impute missing mobility data
    metro_state_df = impute_missing_mobility(metro_state_df)

    county_df_top_metros_remain.to_csv(data/"county_level_policy_evaluation.csv", index=False)

