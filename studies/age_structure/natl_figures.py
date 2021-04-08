import sys
from itertools import chain, islice, product

import adaptive.plots as plt
import geopandas as gpd
import mapclassify
from studies.age_structure.commons import *
from studies.age_structure.epi_simulations import *
from tqdm import tqdm

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

def aggregate_static_percentiles(src, pattern, sum_axis = 0, pct_axis = 0, lim = None):
    total = np.array(0)
    for npz in tqdm(islice(filter(lambda _: "Delhi" not in str(_), src.glob(pattern)), lim)):
        total = total + np.load(npz)['arr_0']
    return np.percentile(total, [50, 5, 95], axis = pct_axis)

def aggregate_dynamic_percentiles(src, pattern, sum_axis = 1, pct_axis = 0, t = 0, lim = None):
    total = np.array(0)
    for npz in tqdm(islice(filter(lambda _: "Delhi" not in str(_), src.glob(pattern)), lim)):
        total = total + np.load(npz)['arr_0'][t].sum(axis = sum_axis)
    return np.percentile(total, [50, 5, 95], axis = pct_axis)

def aggregate_dynamic_percentiles_by_age(src, pattern, sum_axis = 1, pct_axis = 0, t = 0, lim = None):
    total = np.array(0)
    for npz in tqdm(islice(filter(lambda _: "Delhi" not in str(_), src.glob(pattern)), lim)):
        total = total + np.load(npz)['arr_0'][t]
    return np.percentile(total, [50, 5, 95], axis = pct_axis)

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

def plot_state_age_distribution(percentiles, ylabel, fmt, district_spacing = 1.5, age_spacing = 0.1, rotation = 0):
    fig = plt.figure()
    n = len(percentiles)
    state_ordering = list(sorted(
        percentiles.keys(), 
        key = lambda k: percentiles[k][0].max(), 
        reverse = True)
    )
    for (i, state) in enumerate(state_ordering):
        ylls = percentiles[state]
        for j in range(7):
            plt.errorbar(
                x = [district_spacing * i + age_spacing * (j - 3)],
                y = ylls[0, 6-j],
                yerr = [
                    [(ylls[0, 6-j] - ylls[1, 6-j])],
                    [(ylls[2, 6-j] - ylls[0, 6-j])]
                ], 
                fmt = fmt,
                color = agebin_colors[6-j],
                figure = fig,
                label = None if i > 0 else agebin_labels[6-j],
                ms = 12, elinewidth = 5
            )
    plt.xticks(
        [1.5 * _ for _ in range(n)],
        state_ordering,
        rotation = rotation,
        fontsize = "20"
    )
    plt.yticks(fontsize = "20")
    # plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20", ncol = 7, 
    plt.legend(fontsize = "20", ncol = 7, 
        loc = "lower center", bbox_to_anchor = (0.5, 1))
    ymin, ymax = plt.ylim()
    plt.vlines(x = [0.75 + 1.5 * _ for _ in range(n-1)], ymin = ymin, ymax = ymax, color = "gray", alpha = 0.5, linewidths = 2)
    plt.ylim(ymin, ymax)
    plt.gca().grid(False, axis = "x")
    plt.PlotDevice().ylabel(f"{ylabel}\n")

if __name__ == "__main__":
    figs_to_run = set(sys.argv[1:])
    run_all = len(figs_to_run) == 0 # if none specified, run all
    src = fig_src
    params = list(chain([(25, "novax",)], product([25, 50, 100, 200], ["contact", "random", "mortality"])))

    # policy outcomes
    # 2A: deaths
    if "2A" in figs_to_run or "deaths" in figs_to_run or run_all:
        death_percentiles = {
            p: aggregate_static_percentiles(src, f"evaluated_deaths_Maha*{'_'.join(map(str, p))}.npz")
            for p in params 
        }
        outcomes_per_policy(death_percentiles, "deaths", "o", reference = ("no_vax",)) 
        plt.show()

    ## 2B: VSLY
    if "2B" in figs_to_run or "VSLY" in figs_to_run or run_all:
        VSLY_percentiles = {
            # p: aggregate_dynamic_percentiles(src, f"total_VSLY*{'_'.join(map(str, p))}.npz")
            p: aggregate_dynamic_percentiles(src, f"evaluated_VSLY_*{'_'.join(map(str, p))}.npz")
            for p in tqdm(params)
        }
        outcomes_per_policy(
            {k: v * USD/(1e9) for (k, v) in VSLY_percentiles.items()}, "VSLY (USD, billions)", "D",
            reference = (25, "novax")
        ) 
        plt.gca().ticklabel_format(axis = "y", useOffset = False)
        plt.show()

    ## 2C: TEV
    if "2C" in figs_to_run or "TEV" in figs_to_run or "WTP" in figs_to_run or run_all:
        TEV_percentiles = {
            p: aggregate_dynamic_percentiles(src, f"total_TEV*{'_'.join(map(str, p))}.npz")
            for p in tqdm(params)
        }

        outcomes_per_policy({k: v * USD/(1e9) for (k, v) in TEV_percentiles.items()}, "TEV (USD, billions)", "D", reference = (25, "novax")) 
        plt.gca().ticklabel_format(axis = "y", useOffset = False)
        plt.show()

    ## 2D: state x age 
    if "2D" in figs_to_run or "TEV_state_age" in figs_to_run or run_all:
        focus_state_WTP = { 
            state: aggregate_dynamic_percentiles_by_age(src, f"evaluated_WTP_{state}*50_random.npz", sum_axis = 0)
            for state in tqdm(focus_states)
        }
        focus_state_WTP_pc = {
            k: v/districts_to_run.loc[k].filter(regex = "N_[0-9]", axis = 1).sum(axis = 0)[None, :] * USD for (k, v) in focus_state_WTP.items()
        }
        plot_state_age_distribution(focus_state_WTP_pc, "per capita TEV (USD)", "D")
        plt.show()

    # appendix: YLL
    if "YLL" in figs_to_run or run_all:
        YLL_percentiles = {
            p: aggregate_static_percentiles(src, f"evaluated_YLL*Punjab*{'_'.join(map(str, p))}.npz")
            for p in tqdm(params)
        }
        outcomes_per_policy(YLL_percentiles, "YLLs", "o", reference = ("no_vax",))
        plt.show()

    # for state in ["Tamil Nadu"]:#tqdm(districts_to_run.index.get_level_values(0).unique()[1:]):
    #     try: 
    #         plt.figure()
    #         state_WTP_percentiles = {
    #             p: aggregate_dynamic_percentiles(src, f"evaluated_WTP_{state}*{'_'.join(map(str, p))}.npz")
    #             for p in tqdm(params)
    #         }
    #         outcomes_per_policy(
    #             {k: v * USD/(1e9) for (k, v) in state_VSLY_percentiles.items()}, "TEV (USD, billions)", "D",
    #             reference = ("no_vax",)
    #         ) 

    #         outcomes_per_policy(
    #             {k: v * USD/(1e9) for (k, v) in state_VSLY_percentiles.items()}, "VSLY (USD, billions)", "D",
    #             reference = ("no_vax",)
    #         ) 
    #         plt.gca().ticklabel_format(axis = "y", useOffset = False)
    #         plt.title(state, loc = "left")
    #         plt.gcf().set_size_inches((16.8 ,  9.92))
    #         plt.savefig(f"figs/statecheckvsly/VSLY_check_{state}.png")
    #         plt.close("all")


    #         plt.figure()
    #         state_VSLY_percentiles = {
    #             p: aggregate_dynamic_percentiles(src, f"evaluated_VSLY_{state}*{'_'.join(map(str, p))}.npz")
    #             for p in tqdm(params)
    #         }
    #         outcomes_per_policy(
    #             {k: v * USD/(1e9) for (k, v) in state_VSLY_percentiles.items()}, "VSLY (USD, billions)", "D",
    #             reference = (25, "random")
    #         ) 
    #         plt.gca().ticklabel_format(axis = "y", useOffset = False)
    #         plt.show()
    #         plt.title(state, loc = "left")
    #         plt.gcf().set_size_inches((16.8 ,  9.92))
    #         plt.savefig(f"figs/statecheckvsly/VSLY_check_{state}.png")
    #         plt.close("all")

    #         plt.figure()
    #         state_YLL_percentiles = {
    #             p: aggregate_static_percentiles(src, f"evaluated_YLL*_{state}*{'_'.join(map(str, p))}.npz")
    #             for p in tqdm(params)
    #         }
    #         state_YLL_percentiles = {}
    #         for p in tqdm(params):
    #             state_YLL_percentiles[p] = aggregate_static_percentiles(src, f"evaluated_YLLs_{state}*{'_'.join(map(str, p))}.npz")
    #         outcomes_per_policy(state_YLL_percentiles, "YLLs", "o", reference = ("no_vax",))
    #         plt.gca().ticklabel_format(axis = "y", useOffset = False)
    #         plt.show()
    #         plt.title(state, loc = "left")
    #         plt.gcf().set_size_inches((16.8 ,  9.92))
    #         plt.savefig(f"figs/statecheckvsly/YLL_check_{state}.png")
    #         plt.close("all")

        # except Exception as e:
        #     print(state, e)

    # 3C: YLL per million choropleth
    if "3C" in figs_to_run or run_all:
        india = gpd.read_file(data/"india.geojson")\
            .drop(columns = ["id", "dt_code", "st_code", "year"])\
            .rename(columns = {"st_nm": "state"})\
            .set_index(["state", "district"])\
            .rename(index = lambda s: s.replace("and", "And"))\
            .sort_index()
        def load_median_YLL(state_district, phi = 50, vax_policy = "random", src = src):
            state, district = state_district
            try:
                return np.median(np.load(src/f"evaluated_YLL_{state}_{district}_{phi}_{vax_policy}.npz")['arr_0'])
            except FileNotFoundError:
                return np.nan

        districts = districts_to_run.copy()\
            .assign(YLL = districts_to_run.index.map(load_median_YLL))\
            .assign(YLL_per_mn = lambda df: df["YLL"]/(df["N_tot"]/1e6))

        fig, ax = plt.subplots(1, 1)
        scheme = mapclassify.UserDefined(districts.YLL_per_mn, [np.nan, 0, 10, 100, 500, 1000, 5000]) 
        districts["category"] = scheme.yb
        india.join(districts["category"].astype(int)).plot(
            column = "category", 
            linewidth = 0.5,
            edgecolor = "k",
            ax = ax, 
            legend = True,
            categorical = True,
            cmap = "plasma",
            missing_kwds = { 
                "color": "lightgrey",
                "label": "not available"
            }, 
            legend_kwds = { 
                "title": "YLL per million",
                "title_fontsize": "20", 
                "fontsize": "20"
            }
        )
        plt.gca().axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        texts = plt.gca().get_legend().texts
        print(scheme.get_legend_classes())
        print(texts)
        texts[0].set_text("0")
        for i in range(1, 6):
            texts[i].set_text(scheme.get_legend_classes()[i + 1].replace(".00", ""))
        fig.set_size_inches((16.8 ,  9.92*2))
        plt.show()

    # vslyd1Chennai_random, vslyd1Chennai_contact, vslyd1Chennai_mortality = map(lambda p: np.load(p)['arr_0'], src.glob("*VSLYd1*Chennai*200*.npz"))
    # vslyChennai_random, vslyChennai_contact, vslyChennai_mortality = map(lambda p: np.load(p)['arr_0'], src.glob("district_VSLY*Chennai*200*.npz"))
    # vslyndChennai_random, vslyndChennai_contact, vslyndChennai_mortality = map(lambda p: np.load(p)['arr_0'], src.glob("evaluated_VSLYND*Chennai*200*.npz"))
    # vhChennai_random, vhChennai_contact, vhChennai_mortality = map(lambda p: np.load(p)['arr_0'], src.glob("evaluated_vh*Chennai*200*.npz"))
    # deaths_random, deaths_contact, deaths_mortality = map(lambda p: np.load(p)['arr_0'], src.glob("evaluated_deaths*Chennai*200*.npz"))
    
    # deaths_random_inf, deaths_contact_inf, deaths_mortality_inf = map(lambda p: np.load(p)['arr_0'], src.glob("evaluated_deaths*Chennai*1*.npz"))
    # vsly_random_inf, vsly_contact_inf, vsly_mortality_inf = map(lambda p: np.load(p)['arr_0'], src.glob("district_VSLY*Chennai*1*.npz"))

    # 3A: health/consumption
    # if "__3A" in figs_to_run:
    #     summed_wtp_health = np.median(evaluated_WTP_h[state, "50_random"], axis = 0)
    #     summed_wtp_income = np.median(evaluated_WTP_pc[state, "50_random"] - evaluated_WTP_h[state, "50_random"], axis = 0)
    #     plot_component_breakdowns(summed_wtp_health, summed_wtp_income, "health", "consumption", semilogy = True)
    #     plt.show()

    #     # demand curves
    #     demand_curve = get_within_state_wtp_ranking(state, {(k, "50_random"): v for (k, v) in state_WTP_by_district.items()}, 50, "random") 

    #     N_state = districts_to_run.loc[state, :].N_tot.sum()

    # # plot static benchmark
    # figure = plt.figure()
    # x_pop = list(chain(*zip(demand_curve.loc[0]["num_vax"].shift(1).fillna(0), demand_curve.loc[0]["num_vax"])))
    # y_wtp = list(chain(*zip(demand_curve.loc[0]["wtp_pc_usd"], demand_curve.loc[0]["wtp_pc_usd"])))
    # plt.plot(x_pop, y_wtp, figure = figure, color = "grey", linewidth = 2)
    # plt.xticks(fontsize = "20")
    # plt.yticks(fontsize = "20")
    # plt.PlotDevice().ylabel("WTP (USD)\n").xlabel("\nnumber vaccinated")
    # plt.ylim(0, 350)
    # plt.xlim(left = 0, right = N_state)
    # plt.show()

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
