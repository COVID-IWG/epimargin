import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
from epimargin.utils import setup

data, figs = setup()

# get districts comprising each HR 
hom_regions = { 
    "Bangalore": ["Bengaluru", "Bengaluru Rural", "Chikkaballapura", "Kolar", "Ramanagara"],
    "Mysore":    ["Chamarajanagar", "Hassan", "Mandya", "Mysuru", "Tumkur"],
    "Kannada":   ["Dakshina Kannada", "Udupi", "Uttara Kannada"],
    "Belgaum":   ["Belagavi", "Dharwad", "Gadag"],
    "Gulbarga":  ["Ballari", "Kalaburagi", "Koppal", "Raichur"]
}

# set up reverse mapping for groupby 
hom_regions_rev = {v: k for (k, vs) in hom_regions.items() for v in vs}

# set up numeric code <-> HR name mapping
hom_regions_numeric = dict(enumerate(hom_regions.keys(), start=58)) 


# load in sero data 
sero = pd.read_stata("data/kadata.labdate.dta")\
    .drop(columns = ["_merge"])\

sero["S"]  = sero["elisa_pos15"]
sero["t0"] = sero["date_med"]
sero["td"] = sero["t0"] + pd.Timedelta(days = 30)
sero["hr"]  = sero.hom_region.map(hom_regions_numeric)


# pull down COVID 19 India data
paths = {"v3": [data_path(i) for i in (1, 2)], "v4": [data_path(i) for i in range(3, 19)]}
# for target in paths['v3'] + paths['v4']:
#     download_data(data, target)
df = load_all_data(v3_paths = [data/filepath for filepath in paths['v3']],  v4_paths = [data/filepath for filepath in paths['v4']])\
    .query("detected_state == 'Karnataka'")

# get all deaths in KA on Aug 29 by district 
get_time_series(df, "detected_district")\
    .query("status_change_date <= 'Aug 29, 2020'", engine = "python")\
    .Deceased.sum(level = 0)\
    .drop("Other State")\
    .astype(int)\
    .to_csv(data/"ka_cumulative_deaths_aug29.csv")

# aggregate time series by hom_region
df["detected_region"] = df.detected_district.map(hom_regions_rev)
ka_ts = get_time_series(df.dropna(subset = ["detected_region"]), "detected_region").rename(columns = {
    "Deceased":     "dD",
    "Hospitalized": "dT",
    "Recovered":    "dR"
}).unstack(1).fillna(0).stack()

cols = ["dD", "dT", "dR"]
ka_ts_all = pd.concat([ka_ts, ka_ts[cols].cumsum().rename(columns = {col: col[1:] for col in cols})], axis = 1)\
    .drop(columns = ["date", "time", "delta", "logdelta"])\
    .reset_index()\
    .rename(columns = {"detected_region": "hr", "status_change_date": "t0"})

# join sero and case data 
sero = pd.merge(sero, ka_ts_all, left_on = ["hr", "t0"], right_on = ["hr", "t0"])
# add led time series
sero = pd.merge(sero, ka_ts_all, left_on = ["hr", "td"], right_on = ["hr", "t0"], suffixes = ("", "_led"))
# add regional fixed effects dummies, omitting Bangalore
sero = pd.concat([sero, pd.get_dummies(sero.hr, prefix="hom_reg_fe").drop(columns = ["hom_reg_fe_Bangalore"])], axis = 1)


# wheeeee let's run regressions
## set up fixed effects formula
FE = " + urban + " + " + ".join(f"hom_reg_fe_{region}" for region in list(hom_regions.keys())[1:])
total = smf.ols(" T ~ S" + FE, data = sero).fit()
delta = smf.ols("dT ~ S" + FE, data = sero).fit()

print(total.summary())
print(delta.summary())

total_led = smf.ols(" T_led ~ S" + FE, data = sero).fit()
delta_led = smf.ols("dT_led ~ S" + FE, data = sero).fit()

print(total_led.summary())
print(delta_led.summary())
