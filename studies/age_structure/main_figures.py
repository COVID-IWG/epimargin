import adaptive.plots as plt
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *


# plotting functions

def plot_component_breakdowns(color, white, colorlabel, whitelabel, semilogy = False, ylabel = "WTP (USD)"):
    fig, ax = plt.subplots()
    ax.bar(range(7), white * USD, bottom = color * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2, figure = fig)
    ax.bar(range(7), color * USD,                       color = age_group_colors, edgecolor = age_group_colors, linewidth = 2, figure = fig)
    ax.bar(range(7), [0], label = whitelabel, color = "white", edgecolor = "black", linewidth = 2)
    ax.bar(range(7), [0], label = colorlabel, color = "black", edgecolor = "black", linewidth = 2)

    plt.xticks(range(7), age_bin_labels, fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.PlotDevice().ylabel(f"{ylabel}\n")
    if semilogy: plt.semilogy()
    plt.show()

def plot_district_age_distribution(percentiles, ylabel, fmt, N_jk = None, n = 5, district_spacing = 1.5, age_spacing = 0.1, rotation = 0):
    fig = plt.figure()
    district_ordering = list(percentiles.keys())[:n]
    for (i, district) in enumerate(district_ordering):
        ylls = percentiles[district]
        for j in range(7):
            plt.errorbar(
                x = [district_spacing * i + age_spacing * (j - 3)],
                y = ylls[0, 6-j] * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1),
                yerr = [
                    [(ylls[0, 6-j] - ylls[1, 6-j]) * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1)],
                    [(ylls[2, 6-j] - ylls[0, 6-j]) * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1)]
                ], 
                fmt = fmt,
                color = age_group_colors[6-j],
                figure = fig,
                label = None if i > 0 else age_bin_labels[6-j],
                ms = 12, elinewidth = 5
            )
    plt.xticks(
        [1.5 * _ for _ in range(n)],
        district_ordering,
        rotation = rotation,
        fontsize = "20"
    )
    plt.yticks(fontsize = "20")
    plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20", ncol = 7, 
        loc = "lower center", bbox_to_anchor = (0.5, 1))
    ymin, ymax = plt.ylim()
    plt.vlines(x = [0.75 + 1.5 * _ for _ in range(n-1)], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
    plt.ylim(ymin, ymax)
    plt.gca().grid(False, axis = "x")
    plt.PlotDevice().ylabel(f"{ylabel}\n")
    plt.show()


if __name__ == "__main__":
    src = mkdir(data/f"wtp_metrics{num_sims}")

    # aggregate WTP by age
    fig = plt.figure()
    for (i, (md, lo, hi)) in enumerate(zip(*np.percentile(np.sum([v[0] for v in district_WTP.values()], axis = 0), [50, 5, 95], axis = 0))):
        *_, bars = plt.errorbar(x = [i], y = [md * USD], yerr = [[md * USD - lo * USD], [hi * USD - md * USD]], figure = fig,
        fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
        [_.set_alpha(0.5) for _ in bars]
    plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
    plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
    plt.show()

    # health/consumption
    summed_wtp_health = np.median(evaluated_WTP_h[50, "random"], axis = 0)
    summed_wtp_income = np.median(evaluated_WTP_i[50, "random"], axis = 0)
    plot_component_breakdowns(summed_wtp_health, summed_wtp_income, "health", "consumption", semilogy = True)

    # social/private 
    summed_wtp_priv = np.median(evaluated_WTP_p[50, "random"], axis = 0)
    summed_wtp_soc  = np.median(evaluated_WTP_pc[50, "random"] - evaluated_WTP_p[50, "random"], axis = 0)
    plot_component_breakdowns(summed_wtp_soc, summed_wtp_priv, "social", "private", semilogy = False)


    # dist x age 
    per_district_WTP_percentiles = {district: np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for (district, wtp) in district_WTP.items()}
    per_district_YLL_percentiles = {district: np.percentile(yll         , [50, 5, 95], axis = 0) for (district, yll) in district_YLL.items()}
    N_jk_dicts = districts_to_run.filter(like = "N_", axis = 1).to_dict()     
    plot_district_age_distribution(per_district_WTP_percentiles, "per capita WTP (USD)", "D", N_jk = N_jk_dicts)
    plot_district_age_distribution(per_district_YLL_percentiles, "YLL"                 , "o")

#     # death outcomes 
#     #region
#     fig = plt.gcf()

#     md, lo, hi = death_percentiles[(25, "no_vax")]
#     *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]
#     plt.hlines(md, xmin = -1, xmax = 5, linestyles = "dotted", colors = no_vax_color)

#     ##################

#     md, lo, hi = death_percentiles[(25, "contact")]
#     *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(25, "random")]
#     *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(25, "mortality")]
#     *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     ##################

#     md, lo, hi = death_percentiles[(50, "contact")]
#     *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(50, "random")]
#     *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]


#     md, lo, hi = death_percentiles[(50, "mortality")]
#     *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     ##################

#     md, lo, hi = death_percentiles[(100, "contact")]
#     *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(100, "random")]
#     *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(100, "mortality")]
#     *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     # ##################

#     md, lo, hi = death_percentiles[(200, "contact")]
#     *_, bars = plt.errorbar(x = [4 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(200, "random")]
#     *_, bars = plt.errorbar(x = [4], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = death_percentiles[(200, "mortality")]
#     *_, bars = plt.errorbar(x = [4 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]


#     plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
#     plt.xticks([0, 1, 2, 3, 4], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%", "$\phi = 200$%"], fontsize = "20")
#     plt.yticks(fontsize = "20")
#     plt.PlotDevice().ylabel("deaths\n")
#     plt.gca().grid(False, axis = "x")
#     ymin, ymax = plt.ylim()
#     plt.vlines(x = [0.5, 1.5, 2.5, 3.5], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
#     plt.ylim(ymin, ymax)
#     plt.xlim(-0.5, 4.5)
#     plt.show()
#     #endregion

#     # YLL 
#     #region
#     fig = plt.figure()

#     md, lo, hi = YLL_percentiles[(25, "no_vax")]
#     *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]
#     plt.hlines(md, xmin = -1, xmax = 5, linestyles = "dotted", colors = no_vax_color)

#     md, lo, hi = YLL_percentiles[(25, "contact")]
#     *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(25, "random")]
#     *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(25, "mortality")]
#     *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(50, "contact")]
#     *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(50, "random")]
#     *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(50, "mortality")]
#     *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]


#     md, lo, hi = YLL_percentiles[(100, "contact")]
#     *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(100, "random")]
#     *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(100, "mortality")]
#     *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]


#     md, lo, hi = YLL_percentiles[(200, "contact")]
#     *_, bars = plt.errorbar(x = [4 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(200, "random")]
#     *_, bars = plt.errorbar(x = [4], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = YLL_percentiles[(200, "mortality")]
#     *_, bars = plt.errorbar(x = [4 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
#     plt.xticks([0, 1, 2, 3, 4], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%", "$\phi = 200$%"], fontsize = "20")
#     plt.yticks(fontsize = "20")
#     plt.PlotDevice().ylabel("YLLs\n")
#     plt.gca().grid(False, axis = "x")
#     ymin, ymax = plt.ylim()
#     ymin = 4000
#     plt.vlines(x = [0.5, 1.5, 2.5, 3.5], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
#     plt.ylim(ymin, ymax)
#     plt.xlim(-0.5, 4.5)
#     plt.show()
#     #endregion

#     # WTP
#     #region
#     fig = plt.figure()

#     md, lo, hi = WTP_percentiles[(25, "no_vax")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]
#     plt.hlines(md, xmin = -1, xmax = 5, linestyles = "dotted", colors = no_vax_color)

#     md, lo, hi = WTP_percentiles[(25, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(25, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(25, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(50, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(50, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(50, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(100, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(100, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(100, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(200, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(200, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = WTP_percentiles[(200, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
#     plt.xticks([0, 1, 2, 3, 4], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%", "$\phi = 200$%"], fontsize = "20")
#     plt.yticks(fontsize = "20")
#     plt.PlotDevice().ylabel("WTP (USD, billions)\n")
#     plt.gca().grid(False, axis = "x")
#     ymin, ymax = plt.ylim()
#     plt.vlines(x = [0.5, 1.5, 2.5, 3.5], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
#     plt.ylim(ymin, ymax)
#     plt.xlim(-0.5, 4.5)
#     # plt.gca().ticklabel_format(useOffset=False, style='plain')
#     plt.show()
#     #endregion

#     # VSLY 
#     #region
#     fig = plt.figure()

#     md, lo, hi = VSLY_percentiles[(25, "no_vax")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]
#     plt.hlines(md, xmin = -1, xmax = 5, linestyles = "dotted", colors = no_vax_color)

#     md, lo, hi = VSLY_percentiles[(25, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(25, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(25, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(50, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(50, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(50, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(100, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(100, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(100, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(200, "random")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(200, "contact")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     md, lo, hi = VSLY_percentiles[(200, "mortality")] * USD/(1e9)
#     *_, bars = plt.errorbar(x = [4 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#         fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
#     [_.set_alpha(0.5) for _ in bars]

#     plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
#     plt.xticks([0, 1, 2, 3, 4], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%", "$\phi = 200$%"], fontsize = "20")
#     plt.yticks(fontsize = "20")
#     plt.PlotDevice().ylabel("VSLY (USD, billions)\n")
#     plt.gca().grid(False, axis = "x")
#     ymin, ymax = plt.ylim()
#     plt.vlines(x = [0.5, 1.5, 2.5, 3.5], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
#     plt.ylim(ymin, ymax)
#     plt.xlim(-0.5, 4.5)
#     plt.ticklabel_format(style='plain', axis='y', useOffset = False)
#     plt.show()
#     #endregion



# # pd.concat([pd.DataFrame(v).assign(district = k, label = ["median", "percentile05", "percentile95"]) for (k, v) in per_district_WTP_percentiles.items()], axis = 0)\
# #     .rename(columns = dict(enumerate(age_bin_labels)))\
# #     .pipe(lambda df: df[list(df.columns[-2:]) + list(df.columns[:-2])])\
# #     .to_csv("data/rawfigdata/per_district_wtp_percentiles.csv")

# # pd.concat([pd.DataFrame(v).assign(district = k, label = ["median", "percentile05", "percentile95"]) for (k, v) in per_district_YLL_percentiles.items()], axis = 0)\
# #     .rename(columns = dict(enumerate(age_bin_labels)))\
# #     .pipe(lambda df: df[list(df.columns[-2:]) + list(df.columns[:-2])])\
# #     .to_csv("data/rawfigdata/per_district_yll_percentiles.csv")

# def map_pop_dict(agebin, district):
#     i = age_bin_labels.index(agebin)
#     return N_jk_dicts[f"N_{i}"][district]

# # sorted_wtp = pd.concat( 
# #     [ 
# #         pd.DataFrame(v).assign(district = k, label = ["median", "percentile05", "percentile95"])  
# #         for (k, v) in per_district_WTP_percentiles.items() 
# #     ], axis = 0)\
# #     .rename(columns = dict(enumerate(age_bin_labels)))\
# #     .pipe(lambda df: df[list(df.columns[-2:]) + list(df.columns[:-2])])\
# #     .query("label == 'median'")\
# #     .drop(columns = ["label"])\
# #     .set_index("district").stack()\
# #     .sort_values(ascending = False)\
# #     .reset_index()\
# #     .rename(columns = {"level_1": "agebin", 0: "wtp"})

# # sorted_wtp["pop"]    = [map_pop_dict(b, d) for (b, d) in sorted_wtp[["agebin", "district"]].itertuples(index = False)]
# # sorted_wtp["wtp_pc"] = sorted_wtp["wtp"]/sorted_wtp["pop"]
# # sorted_wtp["wtp_pc_usd"] = sorted_wtp["wtp_pc"] * USD
# # sorted_wtp = sorted_wtp.sort_values("wtp_pc", ascending = False)
# # sorted_wtp["num_vax"] = sorted_wtp["pop"].cumsum()

# # x_pop = list(chain(*zip(sorted_wtp["num_vax"].shift(1).fillna(0), sorted_wtp["num_vax"])))
# # y_wtp = list(chain(*zip(sorted_wtp["wtp_pc_usd"], sorted_wtp["wtp_pc_usd"])))
# # plt.plot(x_pop, y_wtp)
# # plt.show()

# # {district: np.median(wtp, axis = 0) for (district, wtp) in district_WTP.items()}

# all_wtp = pd.concat([
#     pd.DataFrame(np.median(v, axis = 1))\
#         .assign(district = k)\
#         .reset_index()\
#         .rename(columns = {"index": "t"})\
#         .rename(columns = dict(enumerate(age_bin_labels)))\
#         .set_index(["t", "district"])
#     for (k, v) in district_WTP.items()
# ], axis = 0)\
#     .stack()\
#     .reset_index()\
#     .rename(columns = {"level_2": "agebin", 0: "agg_wtp"})

# all_wtp["_t"] = -all_wtp["t"]
# all_wtp["pop"]        = [map_pop_dict(b, d) for (b, d) in all_wtp[["agebin", "district"]].itertuples(index = False)]
# all_wtp["wtp_pc"]     = all_wtp["agg_wtp"]/all_wtp["pop"]
# all_wtp["wtp_pc_usd"] = all_wtp["wtp_pc"] * USD 

# all_wtp.sort_values(["_t", "wtp_pc_usd"], ascending = False, inplace = True)
# all_wtp.drop(columns = ["_t"], inplace = True) 
# all_wtp.set_index("t", inplace = True)
# all_wtp["num_vax"] = all_wtp["pop"].groupby(level = 0).cumsum()

# # static optimization
# fig = plt.figure()
# for t in [0, 30, 60, 90, 120]:
#     x_pop = list(chain(*zip(all_wtp.loc[t]["num_vax"].shift(1).fillna(0), all_wtp.loc[t]["num_vax"])))
#     y_wtp = list(chain(*zip(all_wtp.loc[t]["wtp_pc_usd"], all_wtp.loc[t]["wtp_pc_usd"])))
#     if t == 0:
#         x0 = x_pop[:]
#         y0 = y_wtp[:]
#     plt.plot(x_pop, y_wtp, label = t, figure = fig, linewidth = 2)
# plt.legend(title = "starting time", title_fontsize = "24", fontsize = "20")
# plt.xticks(fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
# plt.ylim(0, 350)
# plt.show()

# # dynamic optimization
# daily_doses = 50 * percent * annually * districts_to_run.N_tot.sum()
# distributed_doses = 0
# x_pop = []
# y_wtp = []
# ranking = 0
# for t in range(simulation_range + 1):
#     wtp = all_wtp.loc[t].reset_index()
#     ranking = wtp[(wtp.index >= ranking) & (wtp.num_vax > distributed_doses)].index.min()
#     x_pop += [distributed_doses, distributed_doses + daily_doses]
#     y_wtp += [wtp.iloc[ranking].wtp_pc_usd]*2
#     distributed_doses += daily_doses
#     # print(t, wtp.iloc[ranking], distributed_doses)

# plt.plot(x0, y0, label = "static, t = 0")
# plt.plot(x_pop, y_wtp, label = "dynamic")
# plt.legend(title = "allocation", title_fontsize = "24", fontsize = "20")
# plt.xticks(fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
# plt.ylim(0, 350)
# plt.show()