# preliminary set up
from itertools import cycle
import sys

import epimargin.plots as plt
import numpy as np
import pandas as pd
from epimargin.utils import setup

# don't block plots when running in headless mode in CI
if "headless" in sys.argv:
    sys.argv.remove("headless")
    block_figs = False
else:
    block_figs = True

# set up directories and set plot theme
(data, figs) = setup()
plt.set_theme("minimal")

# download, load, and clean data
from epimargin.etl import download_data
from epimargin.smoothing import notched_smoothing

# a snapshot of this csv is checked into the repo at data/tutorial_timeseries.csv in case you run into download problems
download_data(data, "districts.csv", "https://api.covid19india.org/csv/latest/") 

daily_reports = pd.read_csv(data / "districts.csv", parse_dates = ["Date"])\
    .rename(str.lower, axis = 1)\
    .set_index(["state", "district", "date"])\
    .sort_index()\
    .loc["Maharashtra", "Mumbai"]
daily_cases = daily_reports["confirmed"]\
    .diff()\
    .clip(lower = 0)\
    .dropna()\

smoother = notched_smoothing(window = 5)
smoothed_cases = pd.Series(
    data  = smoother(daily_cases),
    index = daily_cases.index
)

# plot raw and cleaned data 
beg = "December 15, 2020"
end = "March 1, 2021"
training_cases = smoothed_cases[beg:end]

plt.scatter(daily_cases[beg:end].index, daily_cases[beg:end].values, color = "black", s = 5, alpha = 0.5, label = "raw case count data")
plt.plot(training_cases.index, training_cases.values, color = "black", linewidth = 2, label = "notch-filtered, smoothed case count data")
plt.PlotDevice()\
    .l_title("case timeseries for Mumbai")\
    .axis_labels(x = "date", y = "daily cases")\
    .legend()\
    .adjust(bottom = 0.15, left = 0.15)\
    .format_xaxis()\
    .size(9.5, 6)\
    .save(figs / "fig_1.svg")\
    .show(block = block_figs)

# estimate Rt 
from epimargin.estimators import analytical_MPVS

(dates, Rt, Rt_CI_upper, Rt_CI_lower, *_) = analytical_MPVS(training_cases, smoother, infectious_period = 10, totals = False)
plt.Rt(dates[1:], Rt[1:], Rt_CI_upper[1:], Rt_CI_lower[1:], 0.95, legend_loc = "upper left")\
    .l_title("$R_t$ over time for Mumbai")\
    .axis_labels(x = "date", y = "reproductive rate")\
    .adjust(bottom = 0.15, left = 0.15)\
    .size(9.5, 6)\
    .save(figs / "fig_2.svg")\
    .show(block = block_figs)

# set up model
from epimargin.models import SIR

num_sims = 100
N0 = 12.48e6
R0, D0 = daily_reports.loc[end][["recovered", "deceased"]]
I0  = smoothed_cases[:end].sum()
dT0 = smoothed_cases[end]
S0  = N0 - I0 - R0 - D0
Rt0 = Rt[-1] * N0 / S0
no_lockdown = SIR(
    name = "no lockdown", 
    population = N0, 
    dT0 = np.ones(num_sims) * dT0, Rt0 = np.ones(num_sims) * Rt0, I0 = np.ones(num_sims) * I0, R0 = np.ones(num_sims) * R0, D0 = np.ones(num_sims) * D0, S0 = np.ones(num_sims) * S0, infectious_period = 10
)
lockdown = SIR(
    name = "partial lockdown", 
    population = N0, 
    dT0 = np.ones(num_sims) * dT0, Rt0 = np.ones(num_sims) * 0.75 * Rt0, I0 = np.ones(num_sims) * I0, R0 = np.ones(num_sims) * R0, D0 = np.ones(num_sims) * D0, S0 = np.ones(num_sims) * S0, infectious_period = 10
)

# run models forward 
simulation_range = 7
for _ in range(simulation_range):
    lockdown   .parallel_forward_epi_step(num_sims = num_sims)
    no_lockdown.parallel_forward_epi_step(num_sims = num_sims)

# compare policies 
test_cases = smoothed_cases["February 15, 2021":pd.Timestamp(end) + pd.Timedelta(days = simulation_range)]
date_range = pd.date_range(start = end, periods = simulation_range + 1, freq = "D")
legend_entries = [plt.predictions(date_range, model, color) for (model, color) in zip([lockdown, no_lockdown], cycle(plt.SIM_PALETTE))]
train_marker, = plt.plot(test_cases[:end].index, test_cases[:end].values, color = "black")
test_marker,  = plt.plot(test_cases[end:].index, test_cases[end:].values, color = "black", linestyle = "dotted")
markers, _ = zip(*legend_entries)
plt.PlotDevice()\
    .l_title("projected case counts")\
    .axis_labels(x = "date", y = "daily cases")\
    .legend(
        [train_marker, test_marker] + list(markers),
        ["case counts (training)", "case counts (actual)", "case counts (partial lockdown; 95% simulation range)", "case counts (no lockdown; 95% simulation range)"],
        loc = "upper left"
    )\
    .adjust(bottom = 0.15, left = 0.15)\
    .size(9.5, 6)\
    .format_xaxis()\
    .save(figs / "fig_3.svg")\
    .show(block = block_figs)
