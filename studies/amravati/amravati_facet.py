import sys
from itertools import product

import epimargin.plots as plt
import pandas as pd
from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
from epimargin.smoothing import notched_smoothing
from epimargin.utils import cwd

import seaborn as sns

if len(sys.argv) > 1:
    plt.set_theme(sys.argv[1])


# model details
CI        = 0.95
smoothing = 7

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 26)]
}

for target in paths['v3'] + paths['v4']:
    try: 
        download_data(data, target)
    except:
        pass 

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(df, ["detected_state", "detected_district"])

focus = ts.loc[["Maharashtra", "Madhya Pradesh", "Gujarat", "West Bengal", "Tamil Nadu"]]
district_estimates = []

for (state, district) in focus.index.droplevel(-1).unique():
    if district in ["Unknown", "Other State"]:
        continue
    print(state, district)
    try: 
        (
            dates,
            Rt_pred, RR_CI_upper, RR_CI_lower,
            T_pred, T_CI_upper, T_CI_lower,
            total_cases, new_cases_ts,
            anomalies, anomaly_dates
        ) = analytical_MPVS(focus.loc[state, district].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
        district_estimates.append(pd.DataFrame(data = {
            "dates": dates,
            "Rt_pred": Rt_pred,
            "RR_CI_upper": RR_CI_upper,
            "RR_CI_lower": RR_CI_lower,
            "T_pred": T_pred,
            "T_CI_upper": T_CI_upper,
            "T_CI_lower": T_CI_lower,
            "total_cases": total_cases[2:],
            "new_cases_ts": new_cases_ts,
        }).assign(state = state, district = district))
    except Exception as e:
        print(e)

# handle delhi 
delhi_ts = get_time_series(df[df.detected_state == "Delhi"], "detected_state")
(
    dates,
    Rt_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(delhi_ts.Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)

district_estimates.append(pd.DataFrame(data = {
    "dates": dates.get_level_values(1),
    "Rt_pred": Rt_pred,
    "RR_CI_upper": RR_CI_upper,
    "RR_CI_lower": RR_CI_lower,
    "T_pred": T_pred,
    "T_CI_upper": T_CI_upper,
    "T_CI_lower": T_CI_lower,
    "total_cases": total_cases[2:],
    "new_cases_ts": new_cases_ts,
}).assign(state = "Delhi", district = "Delhi"))

district_estimates = pd.concat(district_estimates, axis = 0).set_index(["state", "district"]).sort_index()

coords = { 
    ("Madhya Pradesh", "Khargone")  : (4, 1),
    ("Madhya Pradesh", "Khandwa")   : (5, 1),
    ("Madhya Pradesh", "Betul")     : (6, 1),
    ("Madhya Pradesh", "Chhindwara"): (7, 1),
    ("Gujarat"       , "Surat")     : (1, 2),
    ("Gujarat"       , "Narmada")   : (2, 2),
    ("Maharashtra"   , "Nandurbar") : (3, 2),
    ("Madhya Pradesh", "Barwani")   : (4, 2),
    ("Madhya Pradesh", "Burhanpur") : (5, 2),
    ("Maharashtra"   , "Amravati")  : (6, 2),
    ("Maharashtra"   , "Nagpur")    : (7, 2),
    ("Maharashtra"   , "Dhule")     : (3, 3),
    ("Maharashtra"   , "Jalgaon")   : (4, 3),
    ("Maharashtra"   , "Buldhana")  : (5, 3),
    ("Maharashtra"   , "Akola")     : (6, 3),
    ("Maharashtra"   , "Wardha")    : (7, 3),
    ("Maharashtra"   , "Nashik")    : (3, 4),
    ("Maharashtra"   , "Aurangabad"): (4, 4),
    ("Maharashtra"   , "Jalna")     : (5, 4),
    ("Maharashtra"   , "Washim")    : (6, 4),
    ("Maharashtra"   , "Yavatmal")  : (7, 4),
    ("Maharashtra"   , "Mumbai")    : (1, 5),
    ("Maharashtra"   , "Thane")     : (2, 5),
    ("Maharashtra"   , "Ahmednagar"): (3, 5),
    ("Maharashtra"   , "Parbhani")  : (5, 5),
    ("Maharashtra"   , "Hingoli")   : (6, 5),
    ("Maharashtra"   , "Pune")      : (3, 6),
    ("Maharashtra"   , "Nanded")    : (6, 6),
    ("Delhi"         , "Delhi")     : (8, 2),
    ("West Bengal"   , "Kolkata")   : (8, 3),
    ("Tamil Nadu"    , "Chennai")   : (8, 4),
}

ncols = max(_[0] for _ in coords.values())
nrows = max(_[1] for _ in coords.values())

serodist = pd.read_stata("data/seroprevalence_district.dta")\
    .rename(columns = lambda _:_.replace("_api", ""))\
    .sort_values(["state", "district"])\
    .set_index(["state", "district"])

yticks = {
    "Surat", "Dhule", "Nashik", "Mumbai", "Pune", "Delhi", "Kolkata", "Chennai"
}

xticks = {
    "Surat", "Narmada", "Mumbai", "Thane", "Pune", "Aurangabad", "Parbhani", "Nanded", "Yavatmal", "Chennai"
}

pop_density = pd.read_csv(data/"popdensity.csv").set_index(["state", "district"])
fig, ax_nest = plt.subplots(ncols = ncols, nrows = nrows)
for (j, i) in product(range(nrows), range(ncols)):
    if (i + 1, j + 1) in coords.values():
        continue
    ax_nest[j, i].axis("off")

for ((state, district), (x, y)) in coords.items():
    plt.sca(ax_nest[y - 1, x - 1])
    urban_share = int((1 - serodist.loc[state, ("New " if district == "Delhi" else "") + district]["rural_share"].mean()) * 100)
    density = pop_density.loc[state, district].density
    rt_data = district_estimates.loc[state, district].set_index("dates")["Feb 1, 2021":]
    plt.Rt(rt_data.index, rt_data.Rt_pred, rt_data.RR_CI_upper, rt_data.RR_CI_lower, 0.95, yaxis_colors = False, ymin = 0.5, ymax = 2.0)
    if (x, y) != (4, 1):
        plt.gca().get_legend().remove()
    plt.gca().set_xticks([pd.Timestamp("February 1, 2021"), pd.Timestamp("March 1, 2021"), pd.Timestamp("April 1, 2021")])
    
    plt.PlotDevice()\
        .l_title(district, fontsize = 12)\
        .r_title(f"{urban_share}% urban, {density}/km$^2$", fontsize = 10)

    if district not in xticks:
        plt.gca().set_xticklabels([])
    if district not in yticks:
        plt.gca().set_yticklabels([])

plt.subplots_adjust(hspace = 0.3, wspace = 0.25, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
plt.show()

ridge_districts = [
    "Amravati", 
    "Akola", "Washim",# "Yavatmal", "Wardha", 
    "Nagpur", "Buldhana", 
    #"Hingoli",
    "Parbhani", "Jalgaon", "Jalna",
    # "Dhule", 
    "Aurangabad",
    "Nandurbar", "Nashik", "Ahmednagar",
    "Pune", "Thane", "Mumbai",
    "Delhi", "Kolkata", "Chennai"
]

o = len(ridge_districts) - 3
ridge_states = {o: "Delhi", o + 1: "West Bengal", o + 2: "Tamil Nadu"}
ridge_codes  = {o: "NCT",   o + 1: "WB",          o + 2: "TN"}

fig, axs = plt.subplots(nrows = len(ridge_districts), sharex = True)
for (i, (ax, district)) in enumerate(zip(axs.flat, ridge_districts)):
    state = ridge_states.get(i, "Maharashtra")
    rt_data = district_estimates.loc[state, district]
    rt_data = rt_data[(rt_data.dates >= "Jan 01, 2021") & (rt_data.dates < "Apr 01, 2021")]
    dates, Rt_pred, Rt_CI_upper, Rt_CI_lower = rt_data.dates, rt_data.Rt_pred, rt_data.RR_CI_upper, rt_data.RR_CI_lower
    plt.sca(ax)
    plt.grid(False, which = "both", axis = "both")
    sns.despine(ax = ax, top = True, left = True, right = True)
    plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, 0.95, yaxis_colors = False, ymin = 0.75, ymax = 2.25, legend = i == 0, legend_loc = "upper center", critical_threshold = False)
    # plt.ylabel(district + ", " + ridge_codes.get(i, "MH"), fontsize = 14, fontdict = plt.theme.title, va = "center", ha = "right", rotation = 00, color = "black")
    plt.PlotDevice().l_title("\n" + district + ", " + ridge_codes.get(i, "MH") +"     ", fontsize = 16, ha = "right", va = "top")
    ax.yaxis.tick_right()
    if i == 9:
        ax.yaxis.set_label_position("right")
        plt.PlotDevice().ylabel("reproductive rate ($R_t$)\n\n", rotation = -90, labelpad = 45)
        # .l_title(district, position = (0, 0.90), ha = "left", va = "bottom", fontsize = 12)\

plt.PlotDevice().xlabel("\ndate")
plt.subplots_adjust(hspace = 0.1, left = 0.16, right = 0.90, bottom = 0.07, top = 0.97)
plt.show()
