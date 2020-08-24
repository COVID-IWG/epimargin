from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from etl import *
from adaptive.utils import cwd, days

def plot_rt_interventions(group_name, group, interventions):
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(group["date"], group['RR_pred'], color="slategray")
    plt.fill_between(group['date'].values,group['RR_CI_lower'],group['RR_CI_upper'], alpha = 0.3)
    plt.xticks(rotation=90, )

    for row in interventions.iterrows():
        plt.axvline(x=row[1]["date"], color=row[1]["colour"], linestyle="--")

    lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in interventions["colour"]]
    plt.legend(lines, interventions["intervention"])
    plt.title("Interventions in {}".format(state_name))
    plt.show()

if __name__ == "__main__":

    root = cwd()
    data = root/"data"
    figs = root/"figs"

    case_df = load_us_county_data("covid_confirmed_usafacts.csv")
    case_timeseries = get_case_timeseries(case_df)
    
    us_mobility = load_country_google_mobility("US").rename(columns={"sub_region_1": "state_name", "census_fips_code": "countyfips"})
    interventions = load_intervention_data()
    interventions.columns = [x.replace(" ", "_" ) for x in interventions.columns]

    metro_areas = load_metro_areas(data/"county_metro_state_walk.csv").rename(columns={"state_codes": "state", "county_fips": "countyfips"})
    county_populations = load_us_county_data("covid_county_population_usafacts.csv")
    county_populations = county_populations[county_populations['county_name'] != 'Statewide Unallocated']

    # county level
    county_case_ts = pd.DataFrame(case_timeseries[case_timeseries.index.get_level_values(level=1) != 0])
    county_mobility_ts = us_mobility[~(us_mobility["countyfips"].isna())].set_index(["state_name", "countyfips", "date"])

    county_df = county_case_ts.join(county_mobility_ts).join(county_populations.set_index(['state_name','countyfips'])).reset_index().set_index(["state_name", "countyfips", "date"])
  
    county_interventions = interventions[~(interventions.index.get_level_values(0) == interventions.index.get_level_values(1))]
    county_mask_policy = load_county_mask_data(data/"county_mask_policy.csv")

    county_df = county_interventions.join(county_df, how='right').sort_index()

    county_df = county_df.groupby(['state_name','countyfips']).apply(fill_dummies, list(county_interventions.columns))
    county_df = county_df.join(metro_areas.set_index(["state_name", "countyfips"])["cbsa_fips"])
    county_df["intervention_mask_all_public"] = 0
    county_df = county_df.groupby(['state_name','countyfips']).apply(add_mask_dummies, county_mask_policy)
    county_df = county_df.groupby(["state_name","countyfips"]).apply(add_lag_cols, ["daily_confirmed_cases"])
    
    # only include places once their outbreak has started - seems that 10 cases is the threshold used 
    #county_df = filter_start_outbreak(county_df) - COMMENTED OUT FOR TESTING. UNCOMMENT LATER

    #INCLUDE THIS
    county_df["threshold_ind"] = county_df["daily_confirmed_cases"].apply(lambda x: 1 if x>10 else 0) 

    # filter to top 100 metro areas
    county_df_top_metros = filter_top_metros(county_df)

    # political affiliation
    vote_df = poli_aff(data/"countypres_2000_2016.csv").rename(columns={'state':'state_name'}).set_index(['state_name', 'countyfips'])
    county_df_top_metros = county_df_top_metros.join(vote_df)

    # impute missing mobility data
    county_df_top_metros_remain = impute_missing_mobility(county_df_top_metros)

    county_df_top_metros_remain.to_csv(data/"county_level_policy_evaluation.csv", index=False)

