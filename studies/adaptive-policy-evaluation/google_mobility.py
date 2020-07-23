from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from etl import load_us_county_data, load_country_google_mobility, load_intervention_data, get_case_timeseries, add_lag_cols
from adaptive.utils import cwd, days

def regression_one():

	# results = smf.ols('daily_confirmed_cases ~ retail_and_recreation_percent_change_from_baseline + transit_stations_percent_change_from_baseline + \
	# 				  workplaces_percent_change_from_baseline + residential_percent_change_from_baseline', data=full_df.loc['Illinois','Cook County']).fit()

	# print(results.summary())

	X = cook_county.iloc[:,2:].values

	X = sm.add_constant(X)
	X = sm.add_constant(X, has_constant='add')
	y = cook_county['daily_confirmed_cases'].values
	res = sm.OLS(y, X).fit()
	pass

if __name__ == "__main__":

	root = cwd()
	data = root/"data"

	county_case_df = load_us_county_data("covid_confirmed_usafacts.csv")
	county_case_timeseries = get_case_timeseries(county_case_df)

	county_populations = load_us_county_data("covid_county_population_usafacts.csv")

	county_case_timeseries = county_case_timeseries.reset_index().merge(county_populations, on=["state_name","county_name"]).set_index(["state_name","county_name", "date"])[["cumulative_confirmed_cases","daily_confirmed_cases","population"]]

	us_mobility = load_country_google_mobility("US").rename(columns={"sub_region_1": "state_name", "sub_region_2": "county_name"})
	us_mobility_timeseries = us_mobility.set_index(["state_name", "county_name", "date"])

	full_df = county_case_timeseries.join(us_mobility_timeseries)
	full_df.groupby(["state_name", "county_name"]).apply(add_lag_cols)

	interventions = load_intervention_data()



