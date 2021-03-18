import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.smoothing import notched_smoothing

sns.set_style("whitegrid", {'axes.grid' : False})

smoothed = notched_smoothing(window = 7)

mobility = pd.read_csv("data/2020_IN_Region_Mobility_Report.csv", parse_dates=["date"])
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
    plt.legend(loc = "upper left")
    plt.fill_betweenx((-100, 60), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
    plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = -90, fontdict = plt.note_font, ha = "center", va = "top")
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

plot_mobility(mobility[mobility.sub_region_1.isna()], "India")

plot_mobility(
    mobility[mobility.sub_region_1.isna()], 
    "India", 
    stringency = stringency, 
    annotation = "Google Mobility Data (baseline mobility measured from Jan 3 - Feb 6) + Oxford COVID Policy Tracker",
    until = "Oct 10, 2020")
plt.show()

# mobility vs cases 

from pathlib import Path

import flat_table
from adaptive.etl.commons import download_data

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
plt.plot(df["TT"][:, "delta", "deceased"].index, smoothed(df["TT"][:, "delta", "deceased"].values), label = "Daily Deaths", color = plt.RED)
plt.text(s = "national lockdown", x = pd.to_datetime("April 27, 2020"), y = 200, fontdict = plt.note_font, ha = "center", va = "top")
plt.legend(loc = 'upper left')
plt.ylim(bottom = 0)
lax = plt.gca()
plt.sca(lax.twinx())
plt.plot(df["TT"][:, "delta", "confirmed"].index, smoothed(df["TT"][:, "delta", "confirmed"].values), label = "Daily Cases", color = plt.PRED_PURPLE)
plt.legend(loc = 'upper right')
plt.PlotDevice().ylabel("new cases", rotation = -90, labelpad = 50)
plt.ylim(bottom = 0)
plt.sca(lax)
plt.PlotDevice()\
    .xlabel("\ndate")\
    .ylabel("new deaths\n")
    # .title("\nIndia Case Count and Death Trends")\
    # .annotate("Covid19India.org")\
plt.fill_betweenx(plt.ylim(), pd.to_datetime("March 24, 2020"), pd.to_datetime("June 1, 2020"), color = "black", alpha = 0.05, zorder = -1)
plt.show()
