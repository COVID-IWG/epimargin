from itertools import chain, product

import epimargin.plots as plt
from studies.vaccine_allocation.commons import *
from studies.vaccine_allocation.epi_simulations import *

from studies.vaccine_allocation.natl_figures import aggregate_static_percentiles, outcomes_per_policy, aggregate_dynamic_percentiles

if __name__ == "__main__":
    src = fig_src
    dst = (data/f"../figs/_apr15/state_debug/{experiment_tag}")
    dst.mkdir(exist_ok = True)
    phis = [int(_ * 365 * 100) for _ in phi_points]
    params = list(chain([(phis[0], "novax",)], product(phis, ["contact", "random", "mortality"])))

    for state_code in simulation_initial_conditions.state_code.unique():
        if state_code in ["NL", "SK"]:
            continue
        print(state_code)

        # deaths
        death_percentiles = {
            p: aggregate_static_percentiles(src, f"deaths_{state_code}_*phi{'_'.join(map(str, p))}.npz")
            for p in params 
        }
        outcomes_per_policy(death_percentiles, "deaths", "o", 
            reference = (25, "novax"), 
            phis = [25, 50, 100, 200], 
            vax_policies = ["contact", "random", "mortality"], 
            policy_colors = [contactrate_vax_color, random_vax_color, mortality_vax_color], 
            policy_labels = ["contact rate", "random", "mortality"]
        )
        plt.PlotDevice().l_title(f"{state_code}: deaths")
        plt.gcf().set_size_inches((16.8 ,  9.92))
        plt.savefig(dst / f"{state_code}_deaths.png")
        plt.close("all")

        # vsly 
        VSLY_percentiles = {
            p: aggregate_dynamic_percentiles(src, f"total_VSLY_{state_code}_*phi{'_'.join(map(str, p))}.npz")
            for p in tqdm(params)
        }

        outcomes_per_policy(
            {k: v * USD/(1e9) for (k, v) in VSLY_percentiles.items()}, "VSLY (USD, billions)", "D",
            reference = (25, "novax"), 
            phis = [25, 50, 100, 200], 
            vax_policies = ["contact", "random", "mortality"], 
            policy_colors = [contactrate_vax_color, random_vax_color, mortality_vax_color], 
            policy_labels = ["contact rate", "random", "mortality"]
        )

        plt.PlotDevice().l_title(f"{state_code}: vsly")
        plt.gcf().set_size_inches((16.8 ,  9.92))
        plt.savefig(dst / f"{state_code}_vsly.png")
        plt.close("all")

        # tev 
        TEV_percentiles = {
            p: aggregate_dynamic_percentiles(src, f"total_TEV_{state_code}_*phi{'_'.join(map(str, p))}.npz", drop = ["_SK_", "_NL_"])
            for p in tqdm(params)
        }

        outcomes_per_policy({k: v * USD/(1e9) for (k, v) in TEV_percentiles.items()}, "TEV (USD, billions)", "D", 
            reference = (25, "novax"), 
            phis = [25, 50, 100, 200], 
            vax_policies = ["contact", "random", "mortality"], 
            policy_colors = [contactrate_vax_color, random_vax_color, mortality_vax_color], 
            policy_labels = ["contact rate", "random", "mortality"]
        ) 
        plt.gca().ticklabel_format(axis = "y", useOffset = False)
        plt.gcf().set_size_inches((16.8 ,  9.92))
        plt.PlotDevice().l_title(f"{state_code}: tev")
        plt.savefig(dst / f"{state_code}_tev.png")
        plt.close("all")
