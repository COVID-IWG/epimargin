import pandas as pd

from adaptive.utils import cwd

# define UA/district population ratio cutoff 
pop_ratio_cutoff = 0.5

# use NCT pop for Delhi 
delhi_nct_pop = 16.787e6

# set up data 
root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
}

for target in paths['v3'] + paths['v4']:
    download_data(data, target)

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)

# filter to UA districts, and get time series
dist_UA = pd.read_csv(data/"dist_to_UA_key.csv", names = ["detected_district", "UA", "population_ratio", "UA_population"], header = 0)\
            .dropna()\
            .query(f"population_ratio >= {pop_ratio_cutoff}")\
            .query("UA != 'Delhi'") # deal with Delhi at NCT level 


cases_UA = dist_UA.merge(df)
ts = get_time_series(cases_UA, "detected_district").Hospitalized.unstack().fillna(0).cumsum(axis = 1)

# aggregate districts by UA
UA_ts = {}
for (UA, districts) in dist_UA.groupby("UA")["detected_district"].agg(list).iteritems():
    UA_ts[UA] = ts.loc[[d for d in districts if d in ts.index.values]].sum(axis = 0)

# add back in Delhi, and fix gaps in time series index 
UA_ts["Delhi"] = get_time_series(df.query("detected_state == 'Delhi'")).Hospitalized.cumsum()
UA_ts = pd.DataFrame(data = UA_ts)
UA_ts = UA_ts.reindex(pd.date_range(min(UA_ts.index), max(UA_ts.index)), method = "ffill").fillna(0)
UA_ts["date_start"] = UA_ts.index
UA_ts = UA_ts.unstack()

# add leads
leads = []
populations = {k: int(v.replace(',', '')) for (k, v) in dist_UA[["UA", "UA_population"]].set_index("UA").to_dict()['UA_population'].items()}
populations["Delhi"] = delhi_nct_pop
for UA in UA_ts.index.get_level_values(0).unique():
    print(UA)
    shifted_ts = pd.concat([UA_ts[UA].shift(-_).rename(f"C{_}") for _ in range(11)], axis = 1)
    shifted_ts = pd.concat([shifted_ts] + [shifted_ts.index.shift(-_).to_series().rename(f"t{_}")  for _ in range(11)], axis = 1)
    shifted_ts["population"] = populations[UA]
    shifted_ts["MSA"] = UA
    leads.append(shifted_ts.dropna())

# write out file 
pd.concat(leads)\
    .reset_index()\
    .to_csv(data/"india_UA_covid.csv")