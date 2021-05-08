import epimargin.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epimargin.smoothing import notched_smoothing
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, load_all_data, get_time_series

sns.set_style("whitegrid", {'axes.grid' : False})

smoothed = notched_smoothing(window = 7)

mobility = pd.concat([
    pd.read_csv("data/2020_IN_Region_Mobility_Report.csv", parse_dates=["date"]),
    pd.read_csv("data/2021_IN_Region_Mobility_Report.csv", parse_dates=["date"])
])
stringency = pd.read_csv("data/OxCGRT_latest.csv", parse_dates=["Date"])

def plot_mobility(series, label, stringency = None, until = None, annotation = "Google Mobility Data; baseline mobility measured from Jan 3 - Feb 6"):
    plt.plot(series.date, smoothed(series.retail_and_recreation_percent_change_from_baseline), label = "Retail/Recreation")
    plt.plot(series.date, smoothed(series.grocery_and_pharmacy_percent_change_from_baseline),  label = "Grocery/Pharmacy")
    plt.plot(series.date, smoothed(series.parks_percent_change_from_baseline),                 label = "Parks")
    plt.plot(series.date, smoothed(series.transit_stations_percent_change_from_baseline),      label = "Transit Stations")
    plt.plot(series.date, smoothed(series.workplaces_percent_change_from_baseline),            label = "Workplaces")
    plt.plot(series.date, smoothed(series.residential_percent_change_from_baseline),           label = "Residential")
    if until:
        right = pd.Timestamp(until)
    elif stringency is not None:
        right = stringency.Date.max()
    else:
        right = series.date.iloc[-1]
    lax = plt.gca()
    if stringency is not None: 
        plt.sca(lax.twinx())
        stringency_IN = stringency.query("CountryName == 'India'")
        stringency_US = stringency.query("(CountryName == 'United States') & (RegionName.isnull())", engine = "python")
        plt.plot(stringency_IN.Date, stringency_IN.StringencyIndex, 'k--', alpha = 0.6, label = "IN Measure Stringency")
        plt.plot(stringency_US.Date, stringency_US.StringencyIndex, 'k.' , alpha = 0.6, label = "US Measure Stringency")
        plt.PlotDevice().ylabel("lockdown stringency index", rotation = -90, labelpad = 50)
        plt.legend()
        plt.sca(lax)
    plt.legend(loc = "lower right")
    plt.fill_betweenx((-100, 60), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
    plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = -90, fontdict = plt.theme.note, ha = "center", va = "top")
    plt.PlotDevice()\
        .xlabel("\ndate")\
        .ylabel("% change in mobility\n")
        # .title(f"\n{label}: Mobility & Lockdown Trends")\
        # .annotate(annotation)\
    plt.ylim(-100, 60)

    plt.xlim(left = series.date.iloc[0], right = right)

# for state in ["Bihar", "Maharashtra", "Tamil Nadu", "Punjab", "Delhi"][1:]:
#     plot_mobility(mobility[(mobility.sub_region_1 == state) & (mobility.sub_region_2.isna())], state)

# for city in ["Mumbai", "Bangalore Urban", "Hyderabad"]:
#     plot_mobility(mobility[mobility.sub_region_2 == city], city)

# plot_mobility(mobility[mobility.sub_region_1.isna()], "India")

plot_mobility(
    mobility[mobility.sub_region_1.isna()], 
    "India", 
    stringency = stringency)
plt.PlotDevice()\
    .title("\nIndia: Mobility & Lockdown Trends")\
    .annotate("Google Mobility Data (baseline mobility measured from Jan 3 - Feb 6, 2020) + Oxford COVID Policy Tracker")
plt.show()

# mobility vs cases 

from pathlib import Path

import flat_table
from epimargin.etl.commons import download_data

data = Path("./data")
download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")

# data prep
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)

series = mobility[mobility.sub_region_1.isna()]
plt.plot(series.date, smoothed(series.retail_and_recreation_percent_change_from_baseline), label = "Retail/Recreation")
plt.fill_betweenx((-100, 60), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = -20, fontdict = plt.note_font, ha = "center", va = "top")
plt.ylim(-100, 10)
plt.xlim(series.date.min(), series.date.max())
plt.legend(loc = 'upper right')
lax = plt.gca()
plt.sca(lax.twinx())
plt.plot(df["TT"][:, "delta", "confirmed"].index, smoothed(df["TT"][:, "delta", "confirmed"].values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.legend(loc = 'lower right')
plt.PlotDevice().ylabel("new cases", rotation = -90, labelpad = 50)
plt.sca(lax)
plt.PlotDevice().title("\nIndia Mobility and Case Count Trends")\
    .annotate("Google Mobility Data + Covid19India.org")\
    .xlabel("\ndate")\
    .ylabel("% change in mobility\n")
plt.show()

plt.plot(series.date, smoothed(series.retail_and_recreation_percent_change_from_baseline), label = "Retail/Recreation")
plt.fill_betweenx((-100, 60), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = -20, fontdict = plt.note_font, ha = "center", va = "top")
plt.ylim(-100, 10)
plt.xlim(series.date.min(), series.date.max())
plt.legend(loc = 'upper right')
lax = plt.gca()
plt.sca(lax.twinx())
plt.plot(df["TT"][:, "delta", "confirmed"].index, smoothed(df["TT"][:, "delta", "confirmed"].values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.legend(loc = 'lower right')
plt.PlotDevice().ylabel("new cases", rotation = -90, labelpad = 50)
plt.sca(lax)
plt.PlotDevice().title("\nIndia Mobility and Case Count Trends")\
    .annotate("Google Mobility Data + Covid19India.org")\
    .xlabel("\ndate")\
    .ylabel("% change in mobility\n")
plt.show()


pop_mil = 1370
plt.plot(df["TT"][:, "delta", "tested"].index, smoothed(df["TT"][:, "delta", "tested"]/pop_mil), color = plt.GRN, label = "Daily Tests/Million People") 
plt.PlotDevice()\
    .title("\nIndia Testing Trends")\
    .annotate("COVID19India.org")\
    .xlabel("\ndate")\
    .ylabel("daily tests per million\n")
plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = 100, fontdict = plt.note_font, ha = "center", va = "top")
plt.fill_betweenx((0, 900), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
plt.xlim(df.index.get_level_values(0).min(), df.index.get_level_values(0).max() - pd.Timedelta(days = 7))
plt.ylim(0, 900)
plt.legend(loc = "upper left")
lax = plt.gca()
plt.sca(lax.twinx())
plt.plot(df["TT"][:, "delta", "confirmed"].index, smoothed(df["TT"][:, "delta", "confirmed"].values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.legend(loc = 'upper right')
plt.PlotDevice().ylabel("new cases", rotation = -90, labelpad = 50)
plt.ylim(bottom = 0)
plt.sca(lax)
plt.show()

# cases vs deaths
from pathlib import Path
data = Path("./data")
paths = {"v3": [data_path(i) for i in (1, 2)], "v4": [data_path(i) for i in range(3, 27)]}
for target in paths['v3'] + paths['v4']: 
    download_data(data, target)
df = load_all_data(v3_paths = [data/filepath for filepath in paths['v3']],  v4_paths = [data/filepath for filepath in paths['v4']])\
    .pipe(lambda _: get_time_series(_, ["detected_state"]))\
    .drop(columns = ["date", "time", "delta", "logdelta"])\
    .rename(columns = {
        "Deceased":     "dD",
        "Hospitalized": "dT",
        "Recovered":    "dR"
    }).sum(level = -1).sort_index()

plt.plot(df.index, smoothed(df.dD.values), label = "Daily Deaths", color = plt.RED)
plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = 200, fontdict = plt.theme.note, ha = "center", va = "top")
plt.legend(loc = 'upper left')
plt.ylim(bottom = 0)
lax = plt.gca()
plt.sca(lax.twinx())
plt.plot(df.index, smoothed(df.dT.values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.legend(loc = 'upper right')
plt.PlotDevice().ylabel("new cases", rotation = -90, labelpad = 50)
plt.ylim(bottom = 0)
plt.sca(lax)
plt.PlotDevice()\
    .xlabel("\ndate")\
    .ylabel("new deaths\n")\
    .title("\nIndia Case Count and Death Trends")\
    .annotate("Covid19India.org")
plt.fill_betweenx(plt.ylim(), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
plt.show()


df = df.iloc[:-7]
plt.plot(df.index, smoothed(df.dT.values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.ylim(bottom = 0)
plt.PlotDevice()\
    .axis_labels(x = "date", y = "daily cases")\
    .l_title("\nIndia: case count trends")\
    .r_title("source:\nCovid19India.org")\
    .adjust(bottom = 0.16)
plt.fill_betweenx(plt.ylim(), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
(_, ymax) = plt.ylim()
plt.text(s = "national\nlockdown", x = pd.to_datetime("April 27, 2020"), y = ymax//2, fontdict = plt.theme.note, ha = "center", va = "center")
plt.show()

plt.plot(df.index, smoothed(df.dD.values), label = "Daily Deaths", color = plt.RED)
plt.ylim(bottom = 0)
plt.PlotDevice()\
    .axis_labels(x = "date", y = "daily deaths")\
    .l_title("\nIndia: death count trends")\
    .r_title("source:\nCovid19India.org")\
    .adjust(bottom = 0.16)
plt.fill_betweenx(plt.ylim(), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
(_, ymax) = plt.ylim()
plt.text(s = "national\nlockdown", x = pd.to_datetime("April 27, 2020"), y = ymax//2, fontdict = plt.theme.note, ha = "center", va = "center")
plt.show()