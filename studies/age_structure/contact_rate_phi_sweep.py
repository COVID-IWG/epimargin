from itertools import product

import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import SIR
from adaptive.policy import PrioritizedAssignment, RandomVaccineAssignment, VaccinationPolicy
from studies.age_structure.commons import *

sns.set(style = "whitegrid")

num_sims = 1000

def save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag):
    print(":::: serializing results")

    if "novaccination" in tag:
        pd.DataFrame(model.dT).to_csv(data/f"latest_sims/dT_{tag}.csv")
        pd.DataFrame(model.dD).to_csv(data/f"latest_sims/dD_{tag}.csv")

    # Dead 
    pd.DataFrame((fD * [_.mean() for _ in model.D]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"latest_sims/Dx_{tag}.csv")

# scaling
dT_conf = df[state].loc[:, "delta", "confirmed"] 
dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index)
T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N * seropos)
T_ratio = T_sero/T

print(":: running simulations")

# coefplot metrics
evaluated_deaths = {}
evaluated_YLLs   = {}
ran_models = {}
novax_districts = set()

for (district, seroprevalence, N_district, _, IFR_sero, _) in district_IFR.filter(items = district_codes.keys(), axis = 0).itertuples():
    if district == "Perambalur":
        continue
    # grab timeseries 
    D, R = ts.loc[district][["dD", "dR"]].sum()

    dT_conf_district = ts.loc[district].dT
    dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
    dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
    T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
    T = T_conf_smooth[date]
    T_sero = (N_district * seroprevalence)
    T_ratio = T_sero/T

    T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
    S = N_district - T_scaled

    # run Rt estimation on scaled timeseries 
    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    geo_tag = f"{state}_{district}_"

    dD0 = ts.loc[district].dD.loc[simulation_start]
    I0 = max(0, (T_scaled - R - D))

    for (vax_pct_annual_goal, vax_effectiveness) in product(
        10 ** np.linspace(-1, 1, 11),
        (0.7,)
    ):
        daily_rate = vax_pct_annual_goal/365
        daily_vax_doses = int(daily_rate * N_district)
        daily_net_doses = int(vax_effectiveness * daily_rate * N_district)

        # policies = [
        #     PrioritizedAssignment( 
        #         daily_vax_doses, 
        #         vax_effectiveness, 
        #         np.tile(split_by_age(S), (num_sims, 1)), 
        #         np.tile(np.squeeze(fI * I0), (num_sims, 1)),
        #         IN_age_ratios,
        #         np.array(list(TN_IFRs.values())),
        #         [1, 2, 3, 4, 0, 5, 6], 
        #         "contactrate")
        # ]
        policies = [
            # RandomVaccineAssignment(daily_vax_doses, vax_effectiveness, split_by_age(S), IN_age_ratios), 
            # PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), [6, 5, 4, 3, 2, 1, 0], "mortality"), 
            PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), [1, 2, 3, 4, 0, 5, 6], "contactrate")
        ]

        vaccination_policy: VaccinationPolicy
        for vaccination_policy in policies:
            if vax_pct_annual_goal == 0:
                param_tag = "novaccination"
                if district in novax_districts:
                    break
                else:
                    novax_districts.add(district)
            else: 
                param_tag = "_".join([
                    vaccination_policy.name(),
                    f"ve{int(100*vax_effectiveness)}",
                    f"annualgoal{int(100 * vax_pct_annual_goal)}",
                    f"Rt_threshold{Rt_threshold}"
                ])
            tag = geo_tag + param_tag
            model = SIR(
                name        = district, 
                population  = N_district, 
                dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
                Rt0         = Rt[simulation_start],
                I0          = np.ones(num_sims) * I0, 
                R0          = np.ones(num_sims) * R, 
                D0          = np.ones(num_sims) * D,
                mortality   = (ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
                random_seed = 0
            )
            model.dD[0] = np.ones(num_sims) * dD0

            t = 0
            dVx_adm = [np.zeros(len(IN_age_structure))] # administered doses
            dVx_eff = [np.zeros(len(IN_age_structure))] # effective doses
            dVx_imm = [np.zeros(len(IN_age_structure))] # immunizing doses

            for t in range(1 * 365):
                adm, eff, imm = vaccination_policy.distribute_doses(model, num_sims = num_sims)
                model.m       = vaccination_policy.get_mortality(list(TN_IFRs.values()))
                dVx_adm.append(adm)
                dVx_eff.append(eff)
                dVx_imm.append(imm)
                print("::::", district, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean(), model.S[-1].mean())
                # vaccination_policy.distribute_doses(model, num_sims = num_sims)
                # model.m = vaccination_policy.update_mortality()
                # print("::::", district, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean(), model.S[-1].mean(), model.m.mean())

            save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag)
            
            policy_deaths = np.sum(model.dD, axis = 0) 
            if param_tag in evaluated_deaths:
                evaluated_deaths[param_tag] += policy_deaths
            else:
                evaluated_deaths[param_tag]  = policy_deaths

            # policy_YLL = np.sum(vaccination_policy.dD_bins, axis = 1) @ YLLs
            # if param_tag in evaluated_YLLs:
            #     evaluated_YLLs[param_tag] += policy_YLL
            # else:
            #     evaluated_YLLs[param_tag]  = policy_YLL

evaluated_death_percentiles = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_deaths.items() if "ve70" in k or "novaccination" in k}
contact_percentiles   = {k: v for (k, v) in evaluated_death_percentiles.items() if "contact"   in k}
random_percentiles    = {k: v for (k, v) in evaluated_death_percentiles.items() if "random"    in k}
mortality_percentiles = {k: v for (k, v) in evaluated_death_percentiles.items() if "mortality" in k}
novax_percentiles     = {k: v for (k, v) in evaluated_death_percentiles.items() if "novacc"    in k}

fig = plt.figure()
# plt.errorbar(
#     x = [-1],
#     y = novax_percentiles["novaccination"][1],
#     yerr = [novax_percentiles["novaccination"][1] - [novax_percentiles["novaccination"][0]],[novax_percentiles["novaccination"][2] - novax_percentiles["novaccination"][1]]],
#     fmt = "o",
#     color = no_vax_color,
#     label = "no vaccination",
#     figure = fig
# )

for (dx, (metrics, clr)) in enumerate(zip(
        [contact_percentiles,],# random_percentiles, mortality_percentiles],
        [contactrate_vax_color,]# random_vax_color, mortality_vax_color]
    )):
    for (i, (key, (lo, md, hi))) in enumerate(metrics.items()):
        *_, bars = plt.errorbar(
            x = [i], y = [md], yerr = [[md - lo], [hi - md]],
            figure = fig, fmt = "o", color = clr, label = None if i > 0 else ["contact rate prioritized", "random assignment", "mortality prioritized"][dx],
            ms = 12, elinewidth = 5
        )
        [_.set_alpha(0.5) for _ in bars]
plt.xticks(range(len(metrics)), [f"{phi}%" for phi in (100 * (10 ** np.linspace(-1, 1, 11))).round(0)], fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().ylabel("deaths\n").xlabel("\npercentage of population vaccinated annually")
# plt.ylim(200, 450)
plt.legend(fontsize = "20")
plt.show()


# evaluated_YLL_percentiles = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_YLLs.items() if "ve70" in k or "novaccination" in k}
# contact_percentiles   = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "contact"   in k}
# random_percentiles    = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "random"    in k}
# mortality_percentiles = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "mortality" in k}
# novax_percentiles     = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "novacc"    in k}

# fig = plt.figure()
# plt.errorbar(
#     x = [-1],
#     y = novax_percentiles["novaccination"][1],
#     yerr = [novax_percentiles["novaccination"][1] - [novax_percentiles["novaccination"][0]],[novax_percentiles["novaccination"][2] - novax_percentiles["novaccination"][1]]],
#     fmt = "o",
#     color = no_vax_color,
#     label = "no vaccination",
#     figure = fig
# )

# for (dx, (metrics, clr)) in enumerate(zip(
#         [contact_percentiles, random_percentiles, mortality_percentiles],
#         [contactrate_vax_color, random_vax_color, mortality_vax_color]
#     )):
#     for (i, (key, (lo, md, hi))) in enumerate(metrics.items()):
#         plt.errorbar(
#             x = [i + 0.2*(dx - 1)], y = [md], yerr = [[md - lo], [hi - md]],
#             figure = fig, fmt = "o", color = clr, label = None if i > 0 else ["contact rate prioritized", "random assignment", "mortality prioritized"][dx]
#         )
# plt.xticks(list(range(-1, len(metrics))), [f"$\phi = {phi}$%" for phi in [0, 25, 50, 75, 100, 200, 400]])
# plt.PlotDevice().ylabel("YLLs")
# # plt.ylim(200, 450)
# plt.legend()
# plt.show()