import epimargin.plots as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from epimargin.utils import cwd

from pathlib import Path

# model details
CI        = 0.95
smoothing = 10

plt.set_theme("twitter")

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

lookback = 120
cutoff = 2

state ="Maharashtra"
state_code = "MH"

state_Rt    = pd.read_csv("/Users/satej/Downloads/pipeline_est_MH_state_Rt (1).csv",    parse_dates = ["dates"], index_col = 0)
district_Rt = pd.read_csv("/Users/satej/Downloads/pipeline_est_MH_district_Rt (2).csv", parse_dates = ["dates"], index_col = 0)

latest_Rt = district_Rt[district_Rt.dates == district_Rt.dates.max()].set_index("district")["Rt_pred"].to_dict()

plt.Rt(list(state_Rt.dates), state_Rt.Rt_pred, state_Rt.Rt_CI_lower, state_Rt.Rt_CI_upper, CI)\
    .axis_labels("date", "$R_t$")\
    .title("Maharashtra: $R_t$ over time", ha = "center", x = 0.5)\
    .adjust(left = 0.11, bottom = 0.16)
plt.gcf().set_size_inches(3840/300, 1986/300)
plt.savefig("./MH_Rt_timeseries.png")
plt.clf()

gdf = gpd.read_file("data/maharashtra.json", dpi = 600)

gdf["Rt"] = gdf.district.map(latest_Rt)
fig, ax = plt.subplots()
fig.set_size_inches(3840/300, 1986/300)
plt.choropleth(gdf, title = None, mappable = plt.get_cmap(0.75, 2.5), fig = fig, ax = ax)\
    .adjust(left = 0)
plt.sca(fig.get_axes()[0])
plt.PlotDevice(fig).title(f"{state}: $R_t$ by district", ha = "center", x = 0.5)
plt.axis('off')
plt.savefig(f"./{state_code}_Rt_choropleth.png", dpi = 300)
plt.clf() 

top10 = [(k, "> 3.0" if v > 3 else f"{v:.2f}", v) for (k, v) in sorted(latest_Rt.items(), key = lambda t:t[1], reverse = True)[:10]]

fig, ax = plt.subplots(1,1)
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText = [(k, l) for (k, l, v) in top10], colLabels = ["district", "$R_t$"], loc = 'center', cellLoc = "center")
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if (row == 0):
        cell.set_text_props(fontfamily = plt.theme.label["family"], fontsize = plt.theme.label["size"], fontweight = "semibold")
plt.PlotDevice().title(f"{state}: top districts by $R_t$", ha = "center", x = 0.5)
plt.savefig(f"./{state_code}_Rt_top10.png", dpi = 600)
plt.clf()