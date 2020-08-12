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
    
    us_mobility = load_country_google_mobility("US").rename(columns={"sub_region_1": "state_name", "sub_region_2": "county_name"})
    interventions = load_intervention_data()
    interventions.columns = [x.replace(" ", "_" ) for x in interventions.columns]

    metro_areas = load_metro_areas(data/"county_metro_state_walk.csv").rename(columns={"state_codes": "state", "county_fips": "countyfips"})
    county_populations = load_us_county_data("covid_county_population_usafacts.csv")

    # county level
    county_case_ts = pd.DataFrame(case_timeseries[case_timeseries.index.get_level_values(level=1) != "Statewide Unallocated"])
    county_mobility_ts = us_mobility[~(us_mobility["county_name"].isna())].set_index(["state_name", "county_name", "date"])

    county_df = county_case_ts.join(county_mobility_ts, how="outer").join(county_populations.set_index(['state_name','county_name']), how='left').reset_index().set_index(["state_name", "countyfips", "date"])
    
    county_interventions = interventions[~(interventions.index.get_level_values(0) == interventions.index.get_level_values(1))]
    county_mask_policy = load_county_mask_data(data/"county_mask_policy.csv")

    county_df = county_interventions.join(county_df, how='right').sort_index()
    county_df = county_df.groupby(['state_name','countyfips']).apply(fill_dummies, list(county_interventions.columns))
    county_df = county_df.join(metro_areas.set_index(["state_name", "countyfips"])["cbsa_fips"])
    county_df["intervention_mask_all_public"] = 0
    county_df = county_df.groupby(['state_name','countyfips']).apply(add_mask_dummies, county_mask_policy)
    county_df = county_df.groupby(["state_name","countyfips"]).apply(add_lag_cols, ["daily_confirmed_cases"])
    
    # only include places once their outbreak has started - seems that 10 cases is the threshold used 
    county_df = filter_start_outbreak(county_df)

    county_df_top_metros = filter_top_metros(county_df)

    county_df_top_metros.reset_index().to_csv(data/"county_level_policy_evaluation.csv")

    # state level
    # state_case_ts = case_timeseries.xs("Statewide Unallocated", level=1)
    # rt_estimations = load_rt_estimations(data/"rt_estimations.csv")
    # state_mobility_ts = county_mobility_ts.groupby(["state_name", "date"]).mean() # need to change this to do it proportional to county population
    # state_metros = load_metro_areas(data/"county_metro_state_walk.csv")


    # full_df = county_case_timeseries.join(us_mobility_timeseries)
    # full_df.groupby(["state_name", "county_name"]).apply(add_lag_cols)


    # interventions = add_colours(interventions)
    # interventions.reset_index(inplace=True)

    # for state_code, state in rt_estimations.groupby("state"):
    #     state_name = state_name_lookup[state_code]
    #     state_interventions = interventions[interventions["county_name"] == state_name]
    #     plot_cases_interventions(state_name, state.loc[state_code].reset_index(), state_interventions)



