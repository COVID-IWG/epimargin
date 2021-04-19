from itertools import chain, product
from functools import lru_cache
import epimargin.plots as plt
from studies.age_structure.TN_CMIE.commons import *
from studies.age_structure.TN_CMIE.epi_simulations import *

# data loading
N_jk_dicts = districts_to_run.filter(like = "N_", axis = 1).to_dict()
N_TN       = districts_to_run.N_tot.sum()

def parse_tag(tag):
    return tuple(int(_) if _.isnumeric() else _ for _ in tag.split("_", 1))

def load_metrics(filename):
    npz = np.load(filename)
    return {parse_tag(tag): npz[tag] for tag in npz.files}

def map_pop_dict(agebin, district):
    i = age_bin_labels.index(agebin)
    return N_jk_dicts[f"N_{i}"][district]

def export_WB():
    TN_pop_df = districts_to_run.filter(regex = "N_[0-6]", axis = 1)
    age_dist_wtp = {k1: v * USD/TN_pop_df.loc[k1].values for ((k1, k2, k3), v) in per_district_WTP_percentiles.items() if (k2, k3) == (50, "random")}
    age_dist_wtp_flat = list(chain([k, op, *v[i]] for (i, op) in enumerate(["median", "lo", "hi"]) for (k, v) in age_dist_wtp.items()))

    pd.DataFrame(age_dist_wtp_flat).rename(columns = dict(enumerate(["district", "metric"] + age_bin_labels)))

# calculations
def get_wtp_ranking(district_WTP):
    @lru_cache
    def ranking(phi, vax_policy = "random"):
        all_wtp = pd.concat([
            pd.DataFrame(np.median(v, axis = 1))\
                .assign(district = district)\
                .reset_index()\
                .rename(columns = {"index": "t"})\
                .rename(columns = dict(enumerate(age_bin_labels)))\
                .set_index(["t", "district"])
            for ((district, tag), v) in district_WTP.items() 
            if tag == f"{phi}_{vax_policy}"
        ], axis = 0)\
            .stack()\
            .reset_index()\
            .rename(columns = {"level_2": "agebin", 0: "agg_wtp"})

        all_wtp["_t"] = -all_wtp["t"]
        all_wtp["pop"]        = [map_pop_dict(b, d) for (b, d) in all_wtp[["agebin", "district"]].itertuples(index = False)]
        all_wtp["wtp_pc"]     = all_wtp["agg_wtp"]/all_wtp["pop"]
        all_wtp["wtp_pc_usd"] = all_wtp["wtp_pc"] * USD 

        all_wtp.sort_values(["_t", "wtp_pc_usd"], ascending = False, inplace = True)
        all_wtp.drop(columns = ["_t"], inplace = True) 
        all_wtp.set_index("t", inplace = True)
        all_wtp["num_vax"] = all_wtp["pop"].groupby(level = 0).cumsum()

        return all_wtp
    return ranking

# plotting functions
def outcomes_per_policy(percentiles, metric_label, fmt, 
    phis            = [25, 50, 100, 200], 
    reference       = (25, "no_vax"), 
    reference_color = no_vax_color,
    vax_policies    =  ["contact", "random", "mortality"],
    policy_colors   = [contactrate_vax_color, random_vax_color, mortality_vax_color],
    policy_labels   = ["contact rate priority", "random assignment", "mortality priority"],
    spacing = 0.2):
    fig = plt.figure()

    md, lo, hi = percentiles[reference]
    *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
        fmt = fmt, color = reference_color, label = "no vaccination", ms = 12, elinewidth = 5)
    [_.set_alpha(0.5) for _ in bars]
    plt.hlines(md, xmin = -1, xmax = 5, linestyles = "dotted", colors = reference_color)

    for (i, phi) in enumerate(phis, start = 1):
        for (j, (vax_policy, color, label)) in enumerate(zip(vax_policies, policy_colors, policy_labels)):
            md, lo, hi = death_percentiles[phi, vax_policy]
            *_, bars = plt.errorbar(
                x = [i + spacing * (j - 1)], 
                y = [md], yerr = [[md - lo], [hi - md]], 
                figure = fig,
                fmt = fmt, 
                color = color, 
                label = label if i == 0 else None, 
                ms = 12, elinewidth = 5
            )
            [_.set_alpha(0.5) for _ in bars]

    plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.xticks(range(len(phis) + 1), [f"$\phi = {phi}$%" for phi in ([0] + phis)], fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.PlotDevice().ylabel(f"{metric_label}\n")
    plt.gca().grid(False, axis = "x")
    ymin, ymax = plt.ylim()
    plt.vlines(x = [0.5 + _ for _ in range(len(phis))], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
    plt.ylim(ymin, ymax)
    plt.xlim(-0.5, len(phis) + 1.5)

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

def plot_district_age_distribution(percentiles, ylabel, fmt, phi = 50, vax_policy = "random", N_jk = None, n = 5, district_spacing = 1.5, age_spacing = 0.1, rotation = 0):
    fig = plt.figure()
    district_ordering = list(districts_to_run.index)[:n]
    for (i, district) in enumerate(district_ordering):
        ylls = percentiles[district, phi, vax_policy]
        for j in range(7):
            plt.errorbar(
                x = [district_spacing * i + age_spacing * (j - 3)],
                y = ylls[1, 6-j] * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1),
                yerr = [
                    [(ylls[1, 6-j] - ylls[0, 6-j]) * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1)],
                    [(ylls[2, 6-j] - ylls[1, 6-j]) * USD/(N_jk[f"N_{6-j}"][district] if N_jk else 1)]
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
    plt.PlotDevice().title(f"\n{vax_policy} demand curves").ylabel(f"{ylabel}\n")

def demand_curves(ranking_provider, vax_policy, phis = [25, 50, 100, 200], phi_benchmark = 25, N_state = N_TN):
    wtp_rankings = {phi: ranking_provider(phi, vax_policy) for phi in phis}

    figure = plt.figure()
    lines = []
    
    # benchmark 
    benchmark = wtp_rankings[phi_benchmark]
    x_pop = list(chain(*zip(benchmark.loc[0]["num_vax"].shift(1).fillna(0), benchmark.loc[0]["num_vax"])))
    y_wtp = list(chain(*zip(benchmark.loc[0]["wtp_pc_usd"], benchmark.loc[0]["wtp_pc_usd"])))
    lines.append(plt.plot(x_pop, y_wtp, figure = figure, color = "black", linewidth = 2)[0])
    lines.append(plt.plot(0, 0, color = "white")[0])

    # plot dynamic curve 
    for (phi, all_wtp) in wtp_rankings.items():
        daily_doses = phi * percent * annually * N_state
        distributed_doses = 0
        x_pop = []
        y_wtp = []
        t_vax = []
        ranking = 0
        for t in range(simulation_range):
            wtp = all_wtp.loc[t].reset_index()
            ranking = wtp[(wtp.index >= ranking) & (wtp.num_vax > distributed_doses)].index.min()
            if np.isnan(ranking):
                break
            x_pop += [distributed_doses, distributed_doses + daily_doses]
            t_vax += [t, t+1]
            y_wtp += [wtp.iloc[ranking].wtp_pc_usd]*2
            distributed_doses += daily_doses
        lines.append(
            plt.plot(x_pop, y_wtp, label = f"dynamic, {vax_policy}, $\phi = ${phi}%", figure = figure)[0]
        )
    plt.legend(
        lines,
        [f"static, t = 0, $\phi = ${phi_benchmark}%", ""]  + [f"dynamic, {vax_policy}, $\phi = ${phi}%" for phi in phis],
        title = "allocation", title_fontsize = "24", fontsize = "20")
    plt.xticks(fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
    plt.ylim(0, 350)
    plt.xlim(left = 0, right = N_TN)
    plt.show()

if __name__ == "__main__":
    src = mkdir(data/f"wtp_metrics{num_sims}")

    evaluated_deaths = load_metrics(src/"evaluated_deaths.npz")
    evaluated_YLL    = load_metrics(src/"evaluated_YLL.npz")
    evaluated_WTP    = load_metrics(src/"evaluated_WTP.npz")
    evaluated_VSLY   = load_metrics(src/"evaluated_VSLY.npz")
    evaluated_WTP_h  = load_metrics(src/"evaluated_WTP_h.npz")
    evaluated_WTP_i  = load_metrics(src/"evaluated_WTP_i.npz")
    evaluated_WTP_p  = load_metrics(src/"evaluated_WTP_p.npz")
    evaluated_WTP_pc = load_metrics(src/"evaluated_WTP_pc.npz")
    district_WTP     = load_metrics(src/"district_WTP.npz")
    district_YLL     = load_metrics(src/"district_YLL.npz")

    death_percentiles = {tag: np.percentile(metric,                  [50, 5, 95])           for (tag, metric) in evaluated_deaths.items()}
    YLL_percentiles   = {tag: np.percentile(metric,                  [50, 5, 95])           for (tag, metric) in evaluated_YLL.items()}
    VSLY_percentiles  = {tag: np.percentile(metric[0].sum(axis = 1), [50, 5, 95], axis = 0) for (tag, metric) in evaluated_VSLY.items()}
    WTP_percentiles   = {tag: np.percentile(metric[0].sum(axis = 1), [50, 5, 95], axis = 0) for (tag, metric) in evaluated_WTP.items()}

    # policy outcomes
    # outcomes_per_policy(death_percentiles, "deaths", "o") 
    # plt.show()
    # outcomes_per_policy(YLL_percentiles, "YLLs", "o") 
    # plt.show()
    # outcomes_per_policy(WTP_percentiles, "WTP (USD, billions)", "D") 
    # plt.show()
    # outcomes_per_policy(VSLY_percentiles, "VSLY (USD, billions)", "D") 
    # # plt.gca().ticklabel_format(useOffset = False, style='plain')
    # plt.show()

    # # aggregate WTP by age
    # fig = plt.figure()
    # for (i, (md, lo, hi)) in enumerate(zip(*np.percentile(np.sum([v[0] for v in district_WTP.values()], axis = 0), [50, 5, 95], axis = 0))):
    #     *_, bars = plt.errorbar(x = [i], y = [md * USD], yerr = [[md * USD - lo * USD], [hi * USD - md * USD]], figure = fig,
    #     fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
    #     [_.set_alpha(0.5) for _ in bars]
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
    # plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
    # plt.show()

    # # health/consumption
    summed_wtp_health = np.median(evaluated_WTP_h[50, "random"], axis = 0)
    summed_wtp_income = np.median(evaluated_WTP_pc[50, "random"] - evaluated_WTP_h[50, "random"], axis = 0)
    # summed_wtp_income = np.median(evaluated_WTP_i[50, "random"], axis = 0)
    # plot_component_breakdowns(summed_wtp_health, summed_wtp_income, "health", "consumption", semilogy = True)
    # plt.show()

    # # social/private 
    summed_wtp_priv = np.median(evaluated_WTP_p[50, "random"], axis = 0)
    summed_wtp_soc  = np.median(evaluated_WTP_pc[50, "random"] - evaluated_WTP_p[50, "random"], axis = 0)
    # plot_component_breakdowns(summed_wtp_soc, summed_wtp_priv, "social", "private", semilogy = False)
    # plt.show()

    # # dist x age 
    per_district_WTP_percentiles = {(district, *parse_tag(tag)): np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for ((district, tag), wtp) in district_WTP.items()}
    per_district_YLL_percentiles = {(district, *parse_tag(tag)): np.percentile(yll         , [50, 5, 95], axis = 0) for ((district, tag), yll) in district_YLL.items()}
    
    # plot_district_age_distribution(per_district_WTP_percentiles, "per capita WTP (USD)", "D", N_jk = N_jk_dicts)
    # plt.show()
    # plot_district_age_distribution(per_district_YLL_percentiles, "YLL"                 , "o")
    # plt.show()

    # demand curves
    ranking_provider = get_wtp_ranking(district_WTP)
    demand_curves(ranking_provider, "mortality")
    plt.show()
    demand_curves(ranking_provider, "random")
    plt.show()
    demand_curves(ranking_provider, "contact")
    plt.show()
