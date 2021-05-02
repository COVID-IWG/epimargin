import numpy as np
import pandas as pd
from studies.vaccine_allocation.commons import *

ts = get_TN_timeseries()
df = load_national_timeseries()

# per-capita cases, deaths
## state level 
date_index = pd.date_range(ts.index.get_level_values(1).min(), ts.index.get_level_values(1).max())
percap = ts.loc[list(district_populations.keys())][["dD", "dT"]]\
    .reset_index()
percap["month"] = percap.status_change_date.dt.month.astype(str) + "_" + percap.status_change_date.dt.year.astype(str)
percap["N"] = percap["detected_district"].replace(district_populations)
percap["dD"] = percap["dD"]/percap["N"]
percap["dT"] = percap["dT"]/percap["N"]

percap.groupby(["detected_district", "month"]).apply(np.mean)[["dD", "dT"]]\
    .to_csv(data/"TN_percap.csv")

## national level 
nat_percap = \
df["TT"].loc[:, "delta"].unstack()[["confirmed", "deceased"]]\
    .reset_index()\
    .rename(columns = {"confirmed": "dT", "deceased": "dD", "index": "date"})\
    .assign(month = lambda _: _.date.dt.month.astype(str).str.zfill(2) + "_" + _.date.dt.year.astype(str))\
    .groupby("month")\
    .apply(np.mean)\
    .drop(columns = ["month"])\
    .sort_index()/(1.3e9)
nat_percap.to_csv(data/"IN_percap.csv")


# get levels
## districts
ts.loc[list(district_populations.keys())].cumsum()\
    .rename(columns = lambda x: x[1])\
    .reset_index()\
    .assign(
        month = lambda _: _.status_change_date.dt.month.astype(str).str.zfill(2) + "_" + _.status_change_date.dt.year.astype(str), 
        N = lambda _: _.detected_district.replace(district_populations),
        active_per_cap = lambda _: (_["T"].astype(float) - _["R"].astype(float) - _["D"].astype(float))/_["N"],
        total_deaths_per_cap = lambda _: _["D"].astype(float)/_["N"]
    )\
    .groupby(["detected_district", "month"])\
    .mean()\
    .query("month != '01_2021'")\
    [["active_per_cap", "total_deaths_per_cap"]]\
    .to_csv(data/"TN_levels_percap.csv")

## national
df["TT"].loc[:, "total"].unstack()[["confirmed", "deceased", "recovered"]]\
    .reset_index()\
    .rename(columns = {"index": "date"})\
    .assign(
        month          = lambda _: _.date.dt.month.astype(str).str.zfill(2) + "_" + _.date.dt.year.astype(str),
        active_per_cap = lambda _: (_["confirmed"].astype(float) - _["recovered"].astype(float) - _["deceased"].astype(float))/1.3e9,
        total_deaths_per_cap = lambda _: _["deceased"]/1.3e9
    )\
    .groupby("month")\
    .mean()\
    .drop(columns = ["confirmed", "deceased", "recovered"])\
    .sort_index()\
    .drop(labels = "01_2021")\
    .to_csv(data/"IN_levels_percap.csv")
