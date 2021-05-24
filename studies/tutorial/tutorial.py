# preliminary set up
from pandas.core.indexes.datetimes import date_range
import epimargin.plots as plt
import pandas as pd
import numpy as np
from epimargin.utils import setup

(data, figs) = setup()
plt.set_theme("minimal")

# download, load, and clean data
from epimargin.etl import download_data
# download_data(data, "districts.csv", "https://api.covid19india.org/csv/latest/") # a snapshot of this csv is checked into the repo

daily_reports = pd.read_csv(data / "districts.csv", parse_dates = ["Date"])\
    .rename(str.lower, axis = 1)\
    .set_index(["state", "district", "date"])\
    .sort_index()\
    .loc["Maharashtra", "Mumbai"]
daily_cases = daily_reports["confirmed"]\
    .diff()\
    .clip(lower = 0)\
    .dropna()\

# plot raw and cleaned data 
from epimargin.smoothing import notched_smoothing
beg = "December 1, 2020"
end = "March 1, 2021"
training_cases = daily_cases[beg:end]
smoother = notched_smoothing(window = 10)
smoothed_cases = smoother(training_cases)
# plt.scatter(training_cases.index, training_cases.values, color = "black", s = 5, alpha = 0.5, label = "raw case count data")
# plt.plot(training_cases.index, smoothed_cases, color = "black", linewidth = 2, label = "notch-filtered, smoothed case count data")
# plt.PlotDevice()\
#     .l_title("case timeseries for Mumbai")\
#     .axis_labels(x = "date", y = "daily cases")\
#     .legend()\
#     .adjust(bottom = 0.15, left = 0.15)\
#     .format_xaxis()\
#     .size(9.5, 6)\
#     .save(figs / "fig_1.svg")\
#     .show()

# estimate Rt 
from epimargin.estimators import analytical_MPVS
(dates, Rt, Rt_CI_upper, Rt_CI_lower, *_) =\
    analytical_MPVS(training_cases, smoother, infectious_period = 10, totals = False)
# plt.Rt(dates, Rt, Rt_CI_upper, Rt_CI_lower, 0.95)\
#     .l_title("$R_t$ over time for Mumbai")\
#     .axis_labels(x = "date", y = "reproductive rate")\
#     .adjust(bottom = 0.15, left = 0.15)\
#     .size(9.5, 6)\
#     .save(figs / "fig_2.svg")\
#     .show()

# set up model
from epimargin.models import SIR
num_sims = 100
N0 = 12.48e6
R0, D0 = daily_reports.loc[end][["recovered", "deceased"]]
I0  = smoothed_cases.sum()
dT0 = smoothed_cases[-1]
S0  = N0 - I0 - R0 - D0
Rt0 = Rt[-1] * N0 / S0
no_lockdown = SIR(
    name = "Mumbai (no lockdown)", 
    population = N0, 
    dT0 = np.ones(num_sims) * dT0, Rt0 = np.ones(num_sims) * Rt0, I0 = np.ones(num_sims) * I0, R0 = np.ones(num_sims) * R0, D0 = np.ones(num_sims) * D0, S0 = np.ones(num_sims) * S0, infectious_period = 10
)
lockdown = SIR(
    name = "Mumbai (partial lockdown)", 
    population = N0, 
    dT0 = np.ones(num_sims) * dT0, Rt0 = np.ones(num_sims) * 0.75 * Rt0, I0 = np.ones(num_sims) * I0, R0 = np.ones(num_sims) * R0, D0 = np.ones(num_sims) * D0, S0 = np.ones(num_sims) * S0, infectious_period = 10
)

# run models forward 
for _ in range(14):
    lockdown   .parallel_forward_epi_step(num_sims = num_sims)
    no_lockdown.parallel_forward_epi_step(num_sims = num_sims)

# compare policies 
test_cases = pd.Series(
    data  = smoother(daily_cases[end: "March 15, 2021"]),
    index = pd.date_range(start = end, periods = 15, freq = "D")
)

