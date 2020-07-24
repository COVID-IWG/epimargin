from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf

from etl import load_us_county_data, load_country_google_mobility, load_intervention_data, get_case_timeseries, add_lag_cols
from adaptive.utils import cwd, days

def regression_one():

	# results = smf.ols('daily_confirmed_cases ~ retail_and_recreation_percent_change_from_baseline + transit_stations_percent_change_from_baseline + \
	# 				  workplaces_percent_change_from_baseline + residential_percent_change_from_baseline', data=full_df.loc['Illinois','Cook County']).fit()

	# print(results.summary())

	X = cook_county.iloc[:,2:].values

	X = sm.add_constant(X, has_constant='add')
	y = cook_county['daily_confirmed_cases'].values
	res = sm.OLS(y, X).fit()
	pass

def plot_cases_interventions(group_name, group, interventions):
    fig, ax = plt.subplots(figsize=(12,8))
    group.plot(ax=ax, color="slategray")

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

    county_case_df = load_us_county_data("covid_confirmed_usafacts.csv")
    county_case_timeseries = get_case_timeseries(county_case_df)

    county_populations = load_us_county_data("covid_county_population_usafacts.csv")

    county_case_timeseries = county_case_timeseries.reset_index().merge(county_populations, on=["state_name","county_name"]).set_index(["state_name","county_name", "date"])[["cumulative_confirmed_cases","daily_confirmed_cases","population"]]

    us_mobility = load_country_google_mobility("US").rename(columns={"sub_region_1": "state_name", "sub_region_2": "county_name"})
    us_mobility_timeseries = us_mobility.set_index(["state_name", "county_name", "date"])

    full_df = county_case_timeseries.join(us_mobility_timeseries)
    full_df.groupby(["state_name", "county_name"]).apply(add_lag_cols)

    interventions = load_intervention_data()
    interventions = add_colours(interventions)
    interventions.reset_index(inplace=True)

	state_level = pd.DataFrame(full_df.groupby(["state_name", "date"])["daily_confirmed_cases"].sum())

	for state_name, state in state_level.groupby("state_name"):
		state_interventions = interventions[interventions["county_name"] == state_name]
        plot_cases_interventions(state_name, state.loc[state_name], state_interventions)



