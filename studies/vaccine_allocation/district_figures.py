from itertools import chain, product

import epimargin.plots as plt
from epimargin.etl.covid19india import state_code_lookup
from studies.vaccine_allocation.commons import *
from studies.vaccine_allocation.epi_simulations import *

from studies.vaccine_allocation.natl_figures import aggregate_static_percentiles, outcomes_per_policy, aggregate_dynamic_percentiles

if __name__ == "__main__":
    src = fig_src
    dst0 = (data/f"../figs/_apr15/state_debug/{experiment_tag}").resolve()
    phis = [int(_ * 365 * 100) for _ in phi_points]
    params = list(chain([(phis[0], "novax",)], product(phis, ["contact", "random", "mortality"])))

    for state_code in simulation_initial_conditions.state_code.unique():
        if state_code in ["NL", "SK"]:
            continue
        print(state_code)

        state = state_code_lookup[state_code].replace(" & ", " And ").replace(" and ", " And ")

        dst: Path = dst0 / state_code
        dst.mkdir(exist_ok = True, parents = True)

        for district in simulation_initial_conditions.loc[state].index:
            print(f"  {district}")
            # deaths
            death_percentiles = {
                p: aggregate_static_percentiles(src, f"deaths_{state_code}_{district}*phi{'_'.join(map(str, p))}.npz")
                for p in params 
            }
            outcomes_per_policy(death_percentiles, "deaths", "o", 
                reference = (25, "novax"), 
                phis = [25, 50, 100, 200], 
                vax_policies = ["contact", "random", "mortality"], 
                policy_colors = [contactrate_vax_color, random_vax_color, mortality_vax_color], 
                policy_labels = ["contact rate", "random", "mortality"]
            )
            plt.PlotDevice().l_title(f"{state_code} {district}: deaths")
            plt.gcf().set_size_inches((16.8 ,  9.92))
            plt.savefig(dst / f"{state_code}_{district}_deaths.png")
            plt.close("all")

            # vsly 
            VSLY_percentiles = {
                p: aggregate_dynamic_percentiles(src, f"total_VSLY_{state_code}_{district}_*phi{'_'.join(map(str, p))}.npz")
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
            plt.savefig(dst / f"{state_code}_{district}_vsly.png")
            plt.close("all")

            # tev 
            TEV_percentiles = {
                p: aggregate_dynamic_percentiles(src, f"total_TEV_{state_code}_{district}_*phi{'_'.join(map(str, p))}.npz", drop = ["_SK_", "_NL_"])
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
            plt.savefig(dst / f"{state_code}_{district}_tev.png")
            plt.close("all")

    for (state, code) in [("Tamil Nadu", "TN"), ("Bihar", "BR")]:
        dst = dst0 / code 
        dst.mkdir(exist_ok = True)
        for district in simulation_initial_conditions.query(f"state == '{state}'").index.get_level_values(1).unique():
            cf_consumption = np.load(src / f"c_p0v0{code}_{district}_phi25_novax.npz")['arr_0']
            cons_mean = np.mean(cf_consumption, axis = 1)
            plt.plot(cons_mean)
            plt.PlotDevice().l_title(f"{code} {district}: mean consumption")
            plt.savefig(dst / f"c_p0v0_{district}.png")
            plt.close("all")

            for (phi, pol) in product(phis, ["contact", "random", "mortality"]):
                p_consumption = np.load(src / f"c_p1v1_{code}_{district}_phi{phi}_{pol}.npz")['arr_0']
                cons_mean = np.mean(p_consumption, axis = 1)
                plt.plot(cons_mean)
                plt.PlotDevice().l_title(f"{code} {district}: mean consumption")
                plt.savefig(dst / f"c_p1v1{district}_phi{phi}_{pol}.png")
                plt.close("all")

                qbar = np.load(src / f"q_bar_{code}_{district}_phi{phi}_{pol}.npz")['arr_0']
                qbar_mean = np.mean(qbar, axis = 1)
                plt.plot(qbar_mean)
                plt.PlotDevice().l_title(f"{code} {district}: mean weighted q")
                plt.savefig(dst / f"qbar{district}_phi{phi}_{pol}.png")
                plt.close("all")

        for district in simulation_initial_conditions.query(f"state == '{state}'").index.get_level_values(1).unique():
            dT_cf = np.load(epi_dst / f"{code}_{district}_phi25_novax.npz")['dT']
            dT_random_200 = np.load(epi_dst / f"{code}_{district}_phi200_random.npz")['dT']
            dT_mortality_200 = np.load(epi_dst / f"{code}_{district}_phi200_mortality.npz")['dT']
            plt.plot(np.mean(dT_cf, axis = 1), label = "novax")
            plt.plot(np.mean(dT_cf, axis = 1), label = "random 200")
            plt.plot(np.mean(dT_cf, axis = 1), label = "mortality 200")
            plt.legend()
            plt.PlotDevice().l_title(f"{code} {district}: mean daily cases")
            plt.savefig(dst / f"dT_{district}_phi{phi}_{pol}.png")
            plt.close("all")