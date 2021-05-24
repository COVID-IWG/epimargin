# preliminary set up
import epimargin.plots as plt
import pandas as pd
from epimargin.utils import setup

(data, figs) = setup()
plt.set_theme("minimal")

# download, load, and clean data
from epimargin.etl import download_data

download_data(data, "districts.csv", "https://api.covid19india.org/csv/latest/") # a snapshot of this csv is checked into the repo
daily_cases = pd.read_csv(data / "districts.csv", parse_dates = ["Date"])\
    .rename(str.lower, axis = 1)\
    .set_index(["state", "district", "date"])\
    ["confirmed"]\
    .sort_index()\
    .loc["Maharashtra", "Mumbai"]\
    .diff()\
    .clip(lower = 0)\
    .dropna()\
    ["January 1, 2021":"March 1, 2021"]

# plot raw and cleaned data 
from epimargin.smoothing import notched_smoothing

smoother = notched_smoothing(window = 10)
smoothed_cases = smoother(daily_cases)
plt.scatter(daily_cases.index, daily_cases.values, color = "black", s = 5, alpha = 0.5, label = "raw case count data")
plt.plot(daily_cases.index, smoothed_cases, color = "black", linewidth = 2, label = "notch-filtered, smoothed case count data")
plt.PlotDevice()\
    .l_title("case timeseries for Mumbai")\
    .axis_labels(x = "date", y = "daily cases")\
    .legend()\
    .adjust(bottom = 0.15, left = 0.15)\
    .size(9.5, 6)\
    .save(figs / "fig_1.svg")\
    .show()

# estimate Rt 
from epimargin.estimators import analytical_MPVS
(dates, Rt, Rt_CI_upper, Rt_CI_lower, *_) =\
    analytical_MPVS(daily_cases, smoother, infectious_period = 10, totals = False)
plt.Rt(dates, Rt, Rt_CI_upper, Rt_CI_lower, 0.95)\
    .show()

# set up model

# compare policies 
