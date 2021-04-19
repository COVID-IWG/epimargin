from itertools import chain, product

import epimargin.plots as plt
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *

# data loading
N_jk_dicts = districts_to_run.filter(like = "N_", axis = 1).to_dict()

def parse_tag(tag):
    return tuple(int(_) if _.isnumeric() else _ for _ in tag.split("_", 1))

def load_metrics(filename):
    npz = np.load(filename)
    return {parse_tag(tag): npz[tag] for tag in npz.files}

def map_pop_dict(agebin, district):
    return N_jk_dicts[f"N_{agebin_labels.index(agebin)}"][district]

# calculations
def get_wtp_ranking(district_WTP, phi, vax_policy = "random"):
    all_wtp = pd.concat([
        pd.DataFrame(np.median(v, axis = 1))\
            .assign(district = district)\
            .reset_index()\
            .rename(columns = {"index": "t"})\
            .rename(columns = dict(enumerate(agebin_labels)))\
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

def get_within_state_wtp_ranking(state, district_WTP, phi, vax_policy = "random"):
    all_wtp = pd.concat([
        pd.DataFrame(np.median(v, axis = 1))\
            .assign(district = district)\
            .reset_index()\
            .rename(columns = {"index": "t"})\
            .rename(columns = dict(enumerate(agebin_labels)))\
            .set_index(["t", "district"])
        for ((district, tag), v) in district_WTP.items() 
        if tag == f"{phi}_{vax_policy}"
    ], axis = 0)\
        .stack()\
        .reset_index()\
        .rename(columns = {"level_2": "agebin", 0: "agg_wtp"})

    all_wtp["_t"] = -all_wtp["t"]
    all_wtp["pop"]        = [map_pop_dict(b, (state, d)) for (b, d) in all_wtp[["agebin", "district"]].itertuples(index = False)]
    all_wtp["wtp_pc"]     = all_wtp["agg_wtp"]/all_wtp["pop"]
    all_wtp["wtp_pc_usd"] = all_wtp["wtp_pc"] * USD 

    all_wtp.sort_values(["_t", "wtp_pc_usd"], ascending = False, inplace = True)
    all_wtp.drop(columns = ["_t"], inplace = True) 
    all_wtp.set_index("t", inplace = True)
    all_wtp["num_vax"] = all_wtp["pop"].groupby(level = 0).cumsum()

    return all_wtp

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
            md, lo, hi = percentiles[phi, vax_policy]
            *_, bars = plt.errorbar(
                x = [i + spacing * (j - 1)], 
                y = [md], yerr = [[md - lo], [hi - md]], 
                figure = fig,
                fmt = fmt, 
                color = color, 
                label = label if i == 1 else None, 
                ms = 12, elinewidth = 5,
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
    plt.xlim(-0.5, len(phis) + 0.5)

def plot_component_breakdowns(color, white, colorlabel, whitelabel, semilogy = False, ylabel = "WTP (USD)"):
    fig, ax = plt.subplots()
    ax.bar(range(7), white * USD, bottom = color * USD, color = "white",       edgecolor = agebin_colors, linewidth = 2, figure = fig)
    ax.bar(range(7), color * USD,                       color = agebin_colors, edgecolor = agebin_colors, linewidth = 2, figure = fig)
    ax.bar(range(7), [0], label = whitelabel, color = "white", edgecolor = "black", linewidth = 2)
    ax.bar(range(7), [0], label = colorlabel, color = "black", edgecolor = "black", linewidth = 2)

    plt.xticks(range(7), agebin_labels, fontsize = "20")
    plt.yticks(fontsize = "20")
    plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
    plt.PlotDevice().ylabel(f"{ylabel}\n")
    if semilogy: plt.semilogy()

def plot_district_age_distribution(state, percentiles, ylabel, fmt, phi = 50, vax_policy = "random", N_jk = None, n = 5, district_spacing = 1.5, age_spacing = 0.1, rotation = 0):
    fig = plt.figure()
    district_ordering = list(sorted(
        (k[0] for k in percentiles.keys()), 
        key = lambda k: percentiles[k, phi, vax_policy][0].sum(), 
        reverse = True)
    )[:n]
    for (i, district) in enumerate(district_ordering):
        ylls = percentiles[district, phi, vax_policy]
        for j in range(7):
            plt.errorbar(
                x = [district_spacing * i + age_spacing * (j - 3)],
                y = ylls[0, 6-j] * USD/(N_jk[f"N_{6-j}"][state, district] if N_jk else 1),
                yerr = [
                    [(ylls[0, 6-j] - ylls[1, 6-j]) * USD/(N_jk[f"N_{6-j}"][state, district] if N_jk else 1)],
                    [(ylls[2, 6-j] - ylls[0, 6-j]) * USD/(N_jk[f"N_{6-j}"][state, district] if N_jk else 1)]
                ], 
                fmt = fmt,
                color = agebin_colors[6-j],
                figure = fig,
                label = None if i > 0 else agebin_labels[6-j],
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

if __name__ == "__main__":
    src = mkdir(data/f"focus_wtp_metrics{num_sims}")

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

    all_death_percentiles = {tag: np.percentile(metric,                  [50, 5, 95])           for (tag, metric) in evaluated_deaths.items()}
    all_YLL_percentiles   = {tag: np.percentile(metric,                  [50, 5, 95])           for (tag, metric) in evaluated_YLL.items()}
    all_VSLY_percentiles  = {tag: np.percentile(metric[0].sum(axis = 1), [50, 5, 95], axis = 0) for (tag, metric) in evaluated_VSLY.items()}
    all_WTP_percentiles   = {tag: np.percentile(metric[0].sum(axis = 1), [50, 5, 95], axis = 0) for (tag, metric) in evaluated_WTP.items()}

    for state in focus_states:
        print(state)
        # death_percentiles = {parse_tag(k2): v for ((k1, k2), v) in all_death_percentiles.items() if k1 == state}
        # YLL_percentiles   = {parse_tag(k2): v for ((k1, k2), v) in all_YLL_percentiles  .items() if k1 == state}
        # VSLY_percentiles  = {parse_tag(k2): v for ((k1, k2), v) in all_VSLY_percentiles .items() if k1 == state}
        # WTP_percentiles   = {parse_tag(k2): v for ((k1, k2), v) in all_WTP_percentiles  .items() if k1 == state}
        
        # policy outcomes
        # outcomes_per_policy(death_percentiles, "deaths", "o") 
        # plt.PlotDevice().title(f"{state}: deaths")
        # plt.show()
        # outcomes_per_policy(YLL_percentiles, "YLLs", "o") 
        # plt.PlotDevice().title(f"{state}: YLLs")
        # plt.show()
        # outcomes_per_policy(WTP_percentiles, "WTP (USD, billions)", "D") 
        # plt.PlotDevice().title(f"{state}: willingness to pay")
        # plt.show()
        # outcomes_per_policy(VSLY_percentiles, "VSLY (USD, billions)", "D") 
        # plt.gca().ticklabel_format(axis = "y", useOffset = False)
        # # plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        # plt.PlotDevice().title(f"{state}: VSLY")
        # plt.show()

        # aggregate WTP by age
        # fig = plt.figure()
        # for (i, (md, lo, hi)) in enumerate(zip(*np.percentile(np.sum([v[0] for v in district_WTP.values()], axis = 0), [50, 5, 95], axis = 0))):
        #     *_, bars = plt.errorbar(x = [i], y = [md * USD], yerr = [[md * USD - lo * USD], [hi * USD - md * USD]], figure = fig,
        #     fmt = "D", color = agebin_colors[i], ms = 12, elinewidth = 5, label = agebin_labels[i])
        #     [_.set_alpha(0.5) for _ in bars]
        # plt.xticks([0, 1, 2, 3, 4, 5, 6], agebin_labels, fontsize = "20")
        # plt.yticks(fontsize = "20")
        # plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
        # plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
        # plt.show()

        # health/consumption
        summed_wtp_health = np.median(evaluated_WTP_h[state, "50_random"], axis = 0)
        # summed_wtp_income = np.median(evaluated_WTP_i["50_random"], axis = 0)
        summed_wtp_income = np.median(evaluated_WTP_pc[state, "50_random"] - evaluated_WTP_h[state, "50_random"], axis = 0)
        plot_component_breakdowns(summed_wtp_health, summed_wtp_income, "health", "consumption", semilogy = True)
        plt.show()

        # social/private 
        summed_wtp_priv = np.median(evaluated_WTP_p[state, "50_random"], axis = 0)
        summed_wtp_soc  = np.median(evaluated_WTP_pc[state, "50_random"] - evaluated_WTP_p[state, "50_random"], axis = 0)
        plot_component_breakdowns(summed_wtp_soc, summed_wtp_priv, "social", "private", semilogy = False)
        plt.show()

        # dist x age 
        state_WTP_by_district = {parse_tag(tag)[0]: v for ((s, tag), v) in district_WTP.items() if s == state and tag.endswith("50_random")}
        per_district_WTP_percentiles = {(district, 50, "random"): np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for (district, wtp) in state_WTP_by_district.items()}        
        plot_district_age_distribution(state, per_district_WTP_percentiles, "per capita WTP (USD)", "D", N_jk = N_jk_dicts)
        plt.show()
        # per_district_YLL_percentiles = {(district, *parse_tag(tag)): np.percentile(yll         , [50, 5, 95], axis = 0) for ((district, tag), yll) in district_YLL.items()}
        # plot_district_age_distribution(per_district_YLL_percentiles, "YLL"                 , "o")
        # plt.show()

        # demand curves
        demand_curve = get_within_state_wtp_ranking(state, {(k, "50_random"): v for (k, v) in state_WTP_by_district.items()}, 50, "random") 

        N_state = districts_to_run.loc[state, :].N_tot.sum()

        # plot static benchmark
        figure = plt.figure()
        x_pop = list(chain(*zip(demand_curve.loc[0]["num_vax"].shift(1).fillna(0), demand_curve.loc[0]["num_vax"])))
        y_wtp = list(chain(*zip(demand_curve.loc[0]["wtp_pc_usd"], demand_curve.loc[0]["wtp_pc_usd"])))
        plt.plot(x_pop, y_wtp, figure = figure, color = "grey", linewidth = 2)
        plt.xticks(fontsize = "20")
        plt.yticks(fontsize = "20")
        plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
        plt.ylim(0, 350)
        plt.xlim(left = 0, right = N_state)
        plt.show()

    # lines.append(plt.plot(0, 0, color = "white")[0])

    # # plot dynamic curve 
    # for ((phi, vax_policy), all_wtp) in wtp_rankings.items():
    #     daily_doses = phi * percent * annually * N_TN
    #     distributed_doses = 0
    #     x_pop = []
    #     y_wtp = []
    #     t_vax = []
    #     ranking = 0
    #     for t in range(simulation_range):
    #         wtp = all_wtp.loc[t].reset_index()
    #         ranking = wtp[(wtp.index >= ranking) & (wtp.num_vax > distributed_doses)].index.min()
    #         if np.isnan(ranking):
    #             break
    #         x_pop += [distributed_doses, distributed_doses + daily_doses]
    #         t_vax += [t, t+1]
    #         y_wtp += [wtp.iloc[ranking].wtp_pc_usd]*2
    #         distributed_doses += daily_doses
    #     lines.append(
    #         plt.plot(x_pop, y_wtp, label = f"dynamic, {vax_policy}, $\phi = ${phi}%", figure = figure)[0]
    #     )
    # plt.legend(
    #     lines,
    #     ["static, t = 0, $\phi = $25%", ""]  + [f"dynamic, {vax_policy}, $\phi = ${phi}%" for (phi, vax_policy) in product(phis, ["random", "mortality"])],
    #     title = "allocation", title_fontsize = "24", fontsize = "20")
    # plt.xticks(fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
    # plt.ylim(0, 350)
    # plt.xlim(left = 0, right = N_TN)
    # plt.show()

    # # realized value
    # wtp_ranking_mortality_25  = get_wtp_ranking(district_WTP, 25, "mortality")
    # wtp_ranking_mortality_50  = get_wtp_ranking(district_WTP, 50, "mortality")
    # wtp_ranking_mortality_100 = get_wtp_ranking(district_WTP, 100, "mortality")
    # wtp_ranking_mortality_200 = get_wtp_ranking(district_WTP, 200, "mortality")

    # lines = []

    # figure = plt.figure()
    # x_pop_benchmark = list(chain(*zip(wtp_ranking_mortality_25.loc[0]["num_vax"].shift(1).fillna(0), wtp_ranking_mortality_25.loc[0]["num_vax"])))
    # y_wtp_benchmark = list(chain(*zip(wtp_ranking_mortality_25.loc[0]["wtp_pc_usd"], wtp_ranking_mortality_25.loc[0]["wtp_pc_usd"])))
    # lines.append(plt.plot(x_pop_benchmark, y_wtp_benchmark, figure = figure, color = "black", linewidth = 2)[0])
    # lines.append(plt.plot(0, 0, color = "white")[0])

    # for (phi, ranking) in zip(
    #     [25, 50, 100, 200], 
    #     [wtp_ranking_mortality_25, wtp_ranking_mortality_50, wtp_ranking_mortality_100, wtp_ranking_mortality_200]
    # ):
    #     daily_doses = phi * percent * annually * N_TN
    #     x_pop = []
    #     y_wtp = []
    #     t = 0
    #     for agebin in agebin_labels[::-1]:
    #         t_start = t
    #         N_agebin = ranking.loc[0].query('agebin == @agebin')["pop"].sum()
    #         while (t - t_start) * daily_doses <= N_agebin and t < simulation_range:
    #             age_rankings = ranking.loc[t].query('agebin == @agebin')
    #             avg_wtp = (lambda x: x.values/x.values.sum())(age_rankings["pop"] * daily_doses) @ age_rankings["wtp_pc_usd"]
    #             x_pop += [t*daily_doses, (t+1)*daily_doses]
    #             y_wtp += [avg_wtp]*2
    #             t += 1
    #     lines.append(plt.plot(x_pop, y_wtp, figure = figure)[0])

    # plt.legend(
    #     lines,
    #     ["mortality prioritized demand curve, $t = 0, \phi = 25$%", ""]\
    #      + [f"mortality prioritized, $\phi = ${phi}%" for phi in [25, 50, 100, 200]],
    #     title = "allocation", title_fontsize = "24", fontsize = "20")
    # plt.xticks(fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.PlotDevice().ylabel("per capita social value (USD)\n").xlabel("\nnumber vaccinated")
    # plt.ylim(0, 350)
    # plt.xlim(left = 0, right = N_TN)
    # plt.show()

    # # contact 
    # wtp_ranking_contact_25  = get_wtp_ranking(district_WTP, 25, "contact")
    # wtp_ranking_contact_50  = get_wtp_ranking(district_WTP, 50, "contact")
    # wtp_ranking_contact_100 = get_wtp_ranking(district_WTP, 100, "contact")
    # wtp_ranking_contact_200 = get_wtp_ranking(district_WTP, 200, "contact")

    # lines = []

    # figure = plt.figure()
    # x_pop_benchmark = list(chain(*zip(wtp_ranking_contact_25.loc[0]["num_vax"].shift(1).fillna(0), wtp_ranking_contact_25.loc[0]["num_vax"])))
    # y_wtp_benchmark = list(chain(*zip(wtp_ranking_contact_25.loc[0]["wtp_pc_usd"], wtp_ranking_contact_25.loc[0]["wtp_pc_usd"])))
    # lines.append(plt.plot(x_pop_benchmark, y_wtp_benchmark, figure = figure, color = "black", linewidth = 2)[0])
    # lines.append(plt.plot(0, 0, color = "white")[0])

    # for (phi, ranking) in zip(
    #     [25, 50, 100, 200], 
    #     [wtp_ranking_contact_25, wtp_ranking_contact_50, wtp_ranking_contact_100, wtp_ranking_contact_200]
    # ):
    #     daily_doses = phi * percent * annually * N_TN
    #     x_pop = []
    #     y_wtp = []
    #     t = 0
    #     for agebin in [agebin_labels[_] for _ in [1, 2, 3, 4, 0, 5, 6]]:
    #         t_start = t
    #         N_agebin = ranking.loc[0].query('agebin == @agebin')["pop"].sum()
    #         while (t - t_start) * daily_doses <= N_agebin and t < simulation_range:
    #             age_rankings = ranking.loc[t].query('agebin == @agebin')
    #             avg_wtp = (lambda x: x.values/x.values.sum())(age_rankings["pop"] * daily_doses) @ age_rankings["wtp_pc_usd"]
    #             x_pop += [t*daily_doses, (t+1)*daily_doses]
    #             y_wtp += [avg_wtp]*2
    #             t += 1
    #     lines.append(plt.plot(x_pop, y_wtp, figure = figure)[0])

    # plt.legend(
    #     lines,
    #     ["contact prioritized demand curve, $t = 0, \phi = 25$%", ""]\
    #      + [f"contact prioritized, $\phi = ${phi}%" for phi in [25, 50, 100, 200]],
    #     title = "allocation", title_fontsize = "24", fontsize = "20")
    # plt.xticks(fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.PlotDevice().ylabel("per capita social value (USD)\n").xlabel("\nnumber vaccinated")
    # plt.ylim(0, 350)
    # plt.xlim(left = 0, right = N_TN)
    # plt.show()

    # # random 
    # wtp_ranking_random_25  = get_wtp_ranking(district_WTP, 25, "random")
    # wtp_ranking_random_50  = get_wtp_ranking(district_WTP, 50, "random")
    # wtp_ranking_random_100 = get_wtp_ranking(district_WTP, 100, "random")
    # wtp_ranking_random_200 = get_wtp_ranking(district_WTP, 200, "random")

    # lines = []

    # figure = plt.figure()
    # x_pop_benchmark = list(chain(*zip(wtp_ranking_random_25.loc[0]["num_vax"].shift(1).fillna(0), wtp_ranking_random_25.loc[0]["num_vax"])))
    # y_wtp_benchmark = list(chain(*zip(wtp_ranking_random_25.loc[0]["wtp_pc_usd"], wtp_ranking_random_25.loc[0]["wtp_pc_usd"])))
    # lines.append(plt.plot(x_pop_benchmark, y_wtp_benchmark, figure = figure, color = "black", linewidth = 2)[0])
    # lines.append(plt.plot(0, 0, color = "white")[0])

    # for (phi, ranking) in zip(
    #     [25, 50, 100, 200], 
    #     [wtp_ranking_random_25, wtp_ranking_random_50, wtp_ranking_random_100, wtp_ranking_random_200]
    # ):
    #     daily_doses = phi * percent * annually * N_TN
    #     x_pop = []
    #     y_wtp = []
    #     for t in range(simulation_range):
    #         avg_wtp = (lambda x: x.values/x.values.sum())(ranking.loc[t]["pop"] * daily_doses) @ ranking.loc[t]["wtp_pc_usd"]
    #         x_pop += [t*daily_doses, (t+1)*daily_doses]
    #         y_wtp += [avg_wtp]*2
    #     lines.append(plt.plot(x_pop, y_wtp, figure = figure)[0])

    # plt.legend(
    #     lines,
    #     ["randomly allocated demand curve, $t = 0, \phi = 25$%", ""]\
    #      + [f"randomly allocated, $\phi = ${phi}%" for phi in [25, 50, 100, 200]],
    #     title = "allocation", title_fontsize = "24", fontsize = "20")
    # plt.xticks(fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.PlotDevice().ylabel("per capita social value (USD)\n").xlabel("\nnumber vaccinated")
    # plt.ylim(0, 350)
    # plt.xlim(left = 0, right = N_TN)
    # plt.show()
