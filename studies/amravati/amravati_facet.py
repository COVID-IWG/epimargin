from itertools import product
import pandas as pd

from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data
import adaptive.plots as plt
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd

import seaborn as sns

# model details
CI        = 0.95
smoothing = 14

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 25)]
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

focus = ts.loc[["Maharashtra", "Madhya Pradesh", "Gujarat"]]
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

district_estimates = pd.concat(district_estimates, axis = 0).set_index(["state", "district"])

coords = { 
    ("Madhya Pradesh", "Khandwa")   : (3, 1),
    ("Madhya Pradesh", "Betul")     : (4, 1),
    ("Madhya Pradesh", "Chhindwara"): (5, 1),
    ("Madhya Pradesh", "Seoni")     : (6, 1),
    ("Maharashtra",    "Buldhana")  : (2, 2),
    ("Maharashtra",    "Akola")     : (3, 2),
    ("Maharashtra",    "Amravati")  : (4, 2),
    ("Maharashtra",    "Nagpur")    : (5, 2),
    ("Maharashtra",    "Washim")    : (3, 3),
    ("Maharashtra",    "Yavatmal")  : (4, 3),
    ("Maharashtra",    "Wardha")    : (5, 3),
    ("Maharashtra",    "Mumbai")    : (1, 4),
    ("Maharashtra",    "Pune")      : (2, 5),
}

serodist = pd.read_stata("data/seroprevalence_district.dta")\
    .rename(columns = lambda _:_.replace("_api", ""))\
    .sort_values(["state", "district"])\
    .set_index(["state", "district"])

yticks = {
    "Khandwa", "Buldhana", "Washim", "Mumbai", "Pune"
}

xticks = {"Buldhana", "Washim", "Yavatmal", "Wardha", "Seoni", "Mumbai", "Pune"}

fig, ax_nest = plt.subplots(ncols = 6, nrows = 5)
for (j, i) in product(range(5), range(6)):
    if (i + 1, j + 1) in coords.values():
        continue
    ax_nest[j, i].axis("off")

for ((state, district), (x, y)) in coords.items():
    plt.sca(ax_nest[y - 1, x - 1])
    rural_share = int(serodist.loc[state, district]["rural_share"].mean() * 100)
    rt_data = district_estimates.loc[state, district].set_index("dates")["Feb 1, 2021":]
    plt.Rt(rt_data.index, rt_data.Rt_pred, rt_data.RR_CI_upper, rt_data.RR_CI_lower, 0.95, yaxis_colors = False, ymax = 1.5)
    plt.gca().get_legend().remove()
    plt.gca().set_xticks([pd.Timestamp("February 1, 2021"), pd.Timestamp("March 1, 2021"), pd.Timestamp("April 1, 2021")])
    title = f"{district} ({rural_share}%)"
    if district == "Amravati":
        plt.title(title)
    else:
        plt.title(title, alpha = 0.8)
    if district not in xticks:
        plt.gca().set_xticklabels([])
    if district not in yticks:
        plt.gca().set_yticklabels([])

plt.subplots_adjust(hspace = 0.3, wspace = 0.25, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
plt.show()