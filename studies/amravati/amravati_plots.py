import sys
from itertools import chain

import epimargin.plots as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
from epimargin.smoothing import convolution
from epimargin.utils import cwd
from matplotlib.font_manager import FontProperties

if len(sys.argv) > 1:
    plt.set_theme(sys.argv[1])

root = cwd()
data = root/"data"
figs = root/"figs"

paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 26)]
}

# for target in paths['v3'] + paths['v4']:
#     # try: 
#     #     download_data(data, target)
#     # except:
#     #     pass 

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(df)#, ["detected_state", "detected_district"])

one_day = pd.Timedelta(days = 1)

# fig 1 

infections = ts[ts.date >= "May 01, 2020"].Hospitalized#.sum(level = 2).sort_index()
smoothed  = convolution("uniform")
scatter   = plt.scatter(infections.index[:-7],          infections.values[:-7],  color = "#CC4C75", marker = "s", s = 5, alpha = 0.5)
lineplot, = plt.plot(   infections.index[:-7], smoothed(infections.values[:-7]), color = "#CC4C75", linewidth = 2)
plt.PlotDevice()\
    .l_title("daily confirmed cases in India")\
    .r_title("source:\nCOVID19India")\
    .axis_labels(x = "date", y = "cases", enforce_spacing = True)\
    .adjust(left = 0.10, bottom = 0.15, right = 0.98, top = 0.90)
plt.xlim(infections.index[0] - one_day, infections.index[-1] + one_day)
plt.legend(
    [scatter, lineplot], 
    ["reported infection counts", "7-day moving average"], 
    handlelength = 0.5, framealpha = 0, prop = {'size': 16})
plt.show() 

# fig 2
mob2020 = pd.read_csv("data/2020_IN_Region_Mobility_Report.csv", parse_dates=["date"])
mob2021 = pd.read_csv("data/2021_IN_Region_Mobility_Report.csv", parse_dates=["date"])

mob = pd.concat([mob2020, mob2021]).set_index("date")

IN_mob = mob[mob.sub_region_1.isna()]\
    .filter(like = "_percent_change_from_baseline", axis = 1).mean(axis = 1)
MH_mob = mob[(mob.sub_region_1 == "Maharashtra") & mob.sub_region_2.isna()]\
    .filter(like = "_percent_change_from_baseline", axis = 1).mean(axis = 1)

IN_marker, = plt.plot(IN_mob, color = "dodgerblue")
MH_marker, = plt.plot(MH_mob, color = "darkorange")
plt.PlotDevice()\
    .l_title("mobility trends in India")\
    .r_title("source:\nGoogle Mobility region reports")\
    .axis_labels(x = "date", y = "% change (compared to Jan 2020)", enforce_spacing = True)\
    .adjust(left = 0.10, bottom = 0.15, right = 0.98, top = 0.90)
plt.vlines(x = pd.Timestamp("March 1, 2021"), ymin = -60, ymax = 10, linewidths = 2, color = "grey", linestyles = "dotted")
plt.text(x = pd.Timestamp("March 2, 2021"), y = -40, s = " $2^{nd}$\n wave\n starts", ha = "left", va = "bottom", color = "grey", font_properties = FontProperties(
    family = plt.theme.note["family"], size = 14))
plt.xlim(MH_mob.index[0] - one_day, MH_mob.index[-1] + one_day)
plt.ylim(-60, 10)
plt.legend(
    [IN_marker, MH_marker], 
    ["all India", "Maharashtra"], 
    handlelength = 0.5, framealpha = 0, prop = {'size': 16}, loc = "upper center", ncol = 2)
plt.show()

# fig 5: variants
mutations = pd.read_json(data/"outbreakinfo_mutation_report_data_2021-04-18.json")\
    .set_index("date_time")   \
    .loc["October 1, 2020":]  \
    .drop(columns = "source") \
    * 100

variants = [
    "B.1.617",
    "B.1.1.7",
    "B.1.351",
    "B.1",
    "B.1.1",
    "B.1.1.216",
    "B.1.1.306",
    "B.1.1.345",
    "B.1.36",
    "B.1.36.29",
    "B.1.525",
    "B.1.618",
    "Other",
]
# sns.set(style = "white", palette = sns.color_palette("cool", n_colors = 2 + len(variants)), font = plt.theme.ticks["family"])

sns.set(
    style   = "white", 
    font    = plt.theme.ticks["family"],
    palette = list(chain(*zip(
        sns.color_palette("PuBuGn_r", n_colors = 1 + len(variants)//2), 
        sns.color_palette("YlOrRd_r", n_colors = 1 + len(variants)//2)
    )))
)

# plt.legend(
#     handlelength = 0.6, framealpha = 0, 
#     prop = {'size': 10}, loc = "lower left", 
#     bbox_to_anchor=(0, 0.925), ncol = len(variants), 
#     columnspacing = 1.5, labelspacing = 0.1
# )

plt.stackplot(mutations.index, *[mutations[v] for v in variants], labels = variants, alpha = 0.75)

handles, labels = plt.gca().get_legend_handles_labels()
lgnd = plt.gca().legend(handles[::-1], labels[::-1], 
    handlelength = 0.6, framealpha = 0, 
    prop = {'size': 12}, 
    ncol = 1, loc = "center left", bbox_to_anchor=(1, 0.5),
    handletextpad = 1, labelspacing = 1
)
lgnd.set_title("variant", prop = {"size": 14, "family": plt.theme.label["family"]})
# lgnd._legend_box.align = "left"

plt.xlim(mutations.index[0] - one_day, mutations.index[-1] + one_day)
plt.PlotDevice()\
    .l_title(f"B.1.* prevalence in India")\
    .r_title("source:\nMullen et al. / CVSB / outbreak.info")\
    .axis_labels(x = "date", y = "lineage prevalence in daily samples (%)")\
    .adjust(bottom = 0.15)\
.show()
