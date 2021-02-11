import numpy as np
import pandas as pd
from studies.age_structure.commons import *

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
    .assign(month = lambda _: _.date.dt.month.astype(str) + "_" + _.date.dt.year.astype(str))\
    .groupby("month")\
    .apply(np.mean)\
    .drop(columns = ["month"])\
    .sort_index()/(1.3e9)
nat_percap.to_csv(data/"IN_percap.csv")
