# preliminary set up
import pandas as pd
from epimargin.utils import setup
(data, figs) = setup()

# download, load, and clean data
from epimargin.etl import download_data
download_data(data, "districts.csv", "https://api.covid19india.org/csv/latest/") # a snapshot of this csv is checked into the repo
daily_cases = pd.read_csv(data / "districts.csv", parse_dates = ["date"])\
    .rename(str.lower, axis = 1)\
    .set_index(["state", "district", "date"])\
    ["confirmed"]\
    .sort_index()\
    .loc["Maharashtra", "Mumbai"]\
    .diff()\
    .clip(lower = 0)

# plot raw and cleaned data 
import epimargin.plots as plt 
plt.set_theme("twitter")
plt.plot(
    plt.normalize_dates(daily_cases.index), 
    daily_cases.values
)
plt.PlotDevice().show()

# estimate Rt 

# set up model

# compare policies 