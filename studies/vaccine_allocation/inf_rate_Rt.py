import epimargin.plots as plt
import pandas as pd
from epimargin.estimators import analytical_MPVS
from studies.vaccine_allocation.commons import *

ts = case_death_timeseries(download = False)
district_age_pop = pd.read_csv(data/"all_india_sero_pop.csv").set_index(["state", "district"])


# supplement: Rt distribution

simulation_initial_conditions = pd.read_csv(data/f"all_india_coalesced_initial_conditions{simulation_start.strftime('%b%d')}.csv")\
    .drop(columns = ["Unnamed: 0"])\
    .set_index(["state", "district"])


# fig 1A: infection rates for India and TN
def sero_scaling(sero_pop, ts, survey_date = survey_date):
    R_sero         = (sero_pop.filter(like = "sero", axis = 1).values * sero_pop.filter(regex = "N_[0-6]", axis = 1).values).sum()

    dD_conf        = ts.dD
    dD_conf        = dD_conf.reindex(pd.date_range(dD_conf.index.min(), dD_conf.index.max()), fill_value = 0)
    dD_conf_smooth = pd.Series(smooth(dD_conf), index = dD_conf.index).clip(0).astype(int)
    D_conf_smooth  = dD_conf_smooth.cumsum().astype(int)
    D0             = D_conf_smooth[simulation_start]

    dT_conf        = ts.dT
    dT_conf        = dT_conf.reindex(pd.date_range(dT_conf.index.min(), dT_conf.index.max()), fill_value = 0)
    dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index).clip(0).astype(int)
    T_conf_smooth  = dT_conf_smooth.cumsum().astype(int)
    T_conf         = T_conf_smooth[survey_date if survey_date in T_conf_smooth.index else -1]
    T_sero         = R_sero + D0 
    T_ratio        = T_sero/T_conf

    return (
        T_ratio, 
        T_ratio * dT_conf, 
        T_ratio * dT_conf_smooth
    )

T_ratio_TT, dT_conf_scaled_TT, dT_conf_scaled_smooth_TT = sero_scaling(district_age_pop,                   ts.sum(level = -1))
T_ratio_TN, dT_conf_scaled_TN, dT_conf_scaled_smooth_TN = sero_scaling(district_age_pop.loc["Tamil Nadu"], ts.loc["Tamil Nadu"].sum(level = -1))

N_TT = district_age_pop.N_tot.sum()
N_TN = district_age_pop.loc["Tamil Nadu"].N_tot.sum()

idx = dT_conf_scaled_TT.index[("March 1, 2020" <= dT_conf_scaled_TT.index) & (dT_conf_scaled_TT.index <= simulation_start)]
fig = plt.figure()
scatter_TN = plt.scatter(idx, dT_conf_scaled_TN       [idx]/N_TN, color = TN_color, label = "Tamil Nadu (raw)",      figure = fig, alpha = 0.5, marker = "o", s = 10, zorder = 5)
plot_TN,   = plt.plot   (idx, dT_conf_scaled_smooth_TN[idx]/N_TN, color = TN_color, label = "Tamil Nadu (smoothed)", figure = fig, zorder = 5,  linewidth = 2)
scatter_TT = plt.scatter(idx, dT_conf_scaled_TT       [idx]/N_TT, color = IN_color, label = "India (raw)",           figure = fig, alpha = 0.5, marker = "o", s = 10, zorder = 10)
plot_TT,   = plt.plot   (idx, dT_conf_scaled_smooth_TT[idx]/N_TT, color = IN_color, label = "India (smoothed)",      figure = fig, zorder = 10, linewidth = 2)
plt.xticks(fontsize = "20", rotation = 0)
plt.yticks(fontsize = "20")
plt.legend(
    [scatter_TN, plot_TN, scatter_TT, plot_TT], 
    ["Tamil Nadu (raw)", "Tamil Nadu (smoothed)", "India (raw)", "India (smoothed)"],
    fontsize = "20", ncol = 4,     
    framealpha = 1, handlelength = 0.75,
    loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.gca().xaxis.set_major_formatter(plt.bY_FMT)
plt.gca().xaxis.set_minor_formatter(plt.bY_FMT)
plt.xlim(left = pd.Timestamp("March 1, 2020"), right = pd.Timestamp("April 15, 2021"))
plt.ylim(bottom = 0)
plt.PlotDevice().ylabel("per-capita infection rate\n").xlabel("\ndate")
plt.show()


# 1B: per capita vaccination rates
vax = load_vax_data()\
    .reindex(pd.date_range(start = pd.Timestamp("Jan 1, 2021"), end = simulation_start, freq = "D"), fill_value = 0)\
    [pd.Timestamp("Jan 1, 2021"):simulation_start]\
    .drop(labels = [pd.Timestamp("2021-03-15")]) # handle NAN

plt.plot(vax.index, vax["Tamil Nadu"]/N_TN, color = TN_color, label = "Tamil Nadu", linewidth = 2)
plt.plot(vax.index, vax["Total"]     /N_TT, color = IN_color, label = "India"     , linewidth = 2)
plt.xticks(fontsize = "20", rotation = 0)
plt.yticks(fontsize = "20")
plt.legend(
    fontsize = "20", ncol = 4,     
    framealpha = 1, handlelength = 0.75,
    loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.gca().xaxis.set_major_formatter(plt.DATE_FMT)
plt.gca().xaxis.set_minor_formatter(plt.DATE_FMT)
plt.xlim(left = pd.Timestamp("Jan 1, 2021"), right = pd.Timestamp("April 1, 2021"))
plt.ylim(bottom = 0, top = 0.05)
plt.PlotDevice().ylabel("per-capita vaccination rate\n").xlabel("\ndate")
plt.show()

# fig 1C: probability of death 
dD_TN = np.array(0.0)
for _ in epi_dst.glob("TN*novax.npz"):
    dD_TN = dD_TN + np.diff(np.load(_)['Dj'].sum(axis = -1), axis = 0)

dD_TT = dD_TN.copy()
for _ in filter(lambda _: "TN" not in str(_), epi_dst.glob("novax.npz")):
    dD_TT = dD_TT + np.diff(np.load(_)['Dj'].sum(axis = -1), axis = 0)

percap_death_TN = 100 * np.percentile(dD_TN, [50, 2.5, 97.5], axis = 1)/N_TN
percap_death_TT = 100 * np.percentile(dD_TT, [50, 2.5, 97.5], axis = 1)/N_TT

x = pd.date_range(start = simulation_start, periods = len(percap_death_TN[0]) - 1, freq = "D")

TN_md_marker, = plt.plot(x, percap_death_TN[0][1:], color = TN_color, linewidth = 2, label = "Tamil Nadu (median)")
TN_CI_marker  = plt.fill_between(x, y1 = percap_death_TN[1][1:], y2 = percap_death_TN[2][1:], color = TN_color, alpha = 0.3)

TT_md_marker, = plt.plot(x, percap_death_TT[0][1:], color = IN_color, linewidth = 2, label = "India (median)")
TT_CI_marker  = plt.fill_between(x, y1 = percap_death_TT[1][1:], y2 = percap_death_TT[2][1:], color = IN_color, alpha = 0.3)

plt.legend(
    [(TN_CI_marker, TN_md_marker), (TT_CI_marker, TT_md_marker)], 
    ["Tamil Nadu, median (95% simulation range)", "India, median (95% simulation range)"],
    fontsize = "20", ncol = 2,
    framealpha = 1, handlelength = 0.75,
    loc = "lower center", bbox_to_anchor = (0.5, 1)
)
plt.ylim(bottom = 0)
plt.xlim(left=x[0], right=pd.Timestamp("July 15, 2021"))
plt.gca().xaxis.set_major_formatter(plt.DATE_FMT)
plt.gca().xaxis.set_minor_formatter(plt.DATE_FMT)
plt.xticks(fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().xlabel("\ndate").ylabel("incremental death probability (percentage)\n")
plt.show()


# prob of death by age bin, TN
# epi_src = ext/f"{experiment_tag}_tev_{num_sims}_{simulation_start.strftime('%b%d')}"
dDj_TN = np.array(0.0)
for _ in epi_dst.glob("TN*novax.npz"):
    dDj_TN = dDj_TN + np.diff(np.load(_)['Dj'], axis = 0)
percap_death_j_TN = 100 * np.percentile(dDj_TN, [50, 2.5, 97.5], axis = 1)/\
    district_age_pop.loc["Tamil Nadu"][[f"N_{i}" for i in range(7)]].sum().values

x = pd.date_range(start = simulation_start, periods = 365 - 1, freq = "D")

markers = []
for i in range(7):
    md_marker, = plt.plot(x, percap_death_j_TN[0][1:, i], color = agebin_colors[i], linewidth = 2, label = "Tamil Nadu (median)")
    CI_marker  = plt.fill_between(x, y1 = percap_death_j_TN[1][1:, i], y2 = percap_death_j_TN[2][1:, i], color = agebin_colors[i], alpha = 0.3)
    markers.append((md_marker, CI_marker))
plt.legend(
    markers, agebin_labels,
    fontsize = "20", ncol = 7,
    framealpha = 1, handlelength = 0.75,
    loc = "lower center", bbox_to_anchor = (0.5, 1)
)
plt.ylim(bottom = 0.0000001)
plt.xlim(left=x[0], right=pd.Timestamp("July 15, 2021"))
plt.gca().xaxis.set_major_formatter(plt.DATE_FMT)
plt.gca().xaxis.set_minor_formatter(plt.DATE_FMT)
plt.xticks(fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().xlabel("\ndate").ylabel("incremental death probability (percentage)\n")
# plt.semilogy()
plt.show()

# supplement: Rt distribution (state)
state_ts = ts.sum(level = [0, 2]).sort_index().drop(labels = 
    ["State Unassigned", "Lakshadweep", "Andaman And Nicobar Islands", "Goa"] + 
    ["Sikkim", "Chandigarh", "Mizoram", "Puducherry", "Arunachal Pradesh", 
    "Nagaland", "Manipur", "Meghalaya", "Tripura", "Himachal Pradesh"] + 
    ["Dadra And Nagar Haveli And Daman And Diu"]
)

india_ts = ts.sum(level = -1)

_, Rt_TT, Rt_CI_upper_TT, Rt_CI_lower_TT, *_ =\
    analytical_MPVS(dT_conf_scaled_smooth_TT.loc["Jan 1, 2021":simulation_start], smoothing = lambda _:_, infectious_period = infectious_period, totals = False) 
Rt_TTn, Rt_CI_upper_TTn, Rt_CI_lower_TTn = [np.mean(_[-7:]) for _ in (Rt_TT, Rt_CI_upper_TT, Rt_CI_lower_TT)]

Rt_dist = {}
for state in state_ts.index.get_level_values(0).unique():
    *_, dT_conf_scaled_smooth = sero_scaling(district_age_pop.loc[state], ts.loc[state].sum(level = -1))
    _, Rt, Rt_CI_upper, Rt_CI_lower, *_ =\
        analytical_MPVS(state_ts.loc[state].loc["Jan 1, 2021":simulation_start].dT, smoothing = lambda _:_, infectious_period = infectious_period, totals = False) 
    Rt_dist[state] = [np.mean(_[-7:]) for _ in (Rt, Rt_CI_upper, Rt_CI_lower)]

Rt_dist = {k:v for (k, v) in sorted(Rt_dist.items(), key = lambda e: e[1][0], reverse = True) if v != [0, 0, 0]}

md, hi, lo = map(np.array, list(zip(*Rt_dist.values())))
*_, bars = plt.errorbar(
    # x = [-2] + list(range(len(md))), 
    x = list(range(len(md))), 
    # y = np.r_[Rt_TTn, md], 
    y = md, 
    # yerr = [
    #     np.r_[Rt_TTn - Rt_CI_lower_TTn, md - lo], 
    #     np.r_[Rt_CI_upper_TTn - Rt_TTn, hi - md]
    # ], 
    yerr = [
        md - lo, 
        hi - md
    ], 
    fmt = "s", color = plt.BLK, ms = 8, elinewidth = 10, label = "$R_t$ (95% CI)")
[_.set_alpha(0.3) for _ in bars]
# plt.hlines(Rt_TTn, xmin = -3, xmax = len(md) + 1, linestyles = "dotted", colors = plt.BLK)
# plt.vlines(-1, ymin = 0, ymax = 6, linewidth = 3, colors = "black")
plt.ylim(bottom = 0, top = 6)
# plt.xlim(left = -3, right = len(md))
plt.xlim(left = -1, right = len(md))
plt.PlotDevice().ylabel("reproductive rate ($R_t$)\n").xlabel("\nstate")
# plt.xticks(ticks = [-2] + list(range(len(md))), labels = ["India"] + [state_name_lookup[_][:2] for _ in Rt_dist.keys()], fontsize = "20")
plt.xticks(list(range(len(md))), labels = [state_name_lookup[_][:2] for _ in Rt_dist.keys()], fontsize = "20")
plt.yticks(fontsize = "20")
plt.subplots_adjust(left = 0.06, right = 0.94)
plt.gca().grid(False, axis = "y")
plt.legend(fontsize = "20")
plt.gcf().set_size_inches((16.8 * 2,  9.92))
plt.show()

# supplement: Rt distribution (district)
sic = simulation_initial_conditions.sort_values("Rt", ascending = False)
*_, bars = plt.errorbar(x = range(len(sic)), y = sic.Rt, yerr = [sic.Rt - sic.Rt_lower, sic.Rt_upper - sic.Rt], fmt = "s", color = plt.BLK, ms = 2)
for _ in bars: _.set_alpha(0.3)
plt.xlim(0, len(sic))
plt.ylim(0, 8)
plt.subplots_adjust(left = 0.02, bottom = 0.02, right = 0.98, top = 0.98)
plt.show()
