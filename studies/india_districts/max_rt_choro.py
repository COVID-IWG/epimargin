import epimargin.plots as plt
import pandas as pd
import geopandas as gpd 

rt  = pd.read_csv("data/india_states_max_Rt.csv")
rt.state = rt.state.str.replace("&", "and")
gdf = gpd.read_file("data/india.json").dissolve("st_nm")

mappable = plt.get_cmap(1, 4, "viridis")
fig, ax = plt.subplots()
gdf["pt"] = gdf["geometry"].centroid
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title(title, loc="left", fontdict=label_font) 
# gdf = gdf.merge(rt, left_on = "st_nm", right_on = "state")
gdf.plot(color=[mappable.to_rgba(_) for _ in gdf["max_Rt"]], ax = ax, edgecolors="black", linewidth=0.5, missing_kwds = {"color": "dimgray", "edgecolor": "white"})

for (_, row) in gdf.iterrows():
    label = label_fn(row)
    a1 = ax.annotate(s=f"{label}{Rt_c}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = note_font["family"], color="white", **label_kwargs)
    a1.set_path_effects([Stroke(linewidth = 2, foreground = "black"), Normal()])
cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
cb = fig.colorbar(mappable = mappable, orientation = "vertical", cax = cbar_ax)
cbar_ax.set_title("$R_t$", fontdict = plt.note_font)
