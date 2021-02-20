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

num_sims = 10000

def save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag):
    print(":::: serializing results")

    # Susceptibles
    pd.DataFrame((fS * [_.mean() for _ in model.S]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"cc100/Sx_{tag}.csv")

    # Infectious 
    pd.DataFrame((fI * [_.mean() for _ in model.I]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"cc100/Ix_{tag}.csv")

    # Recovered
    pd.DataFrame((fR * [_.mean() for _ in model.R]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"cc100/Rx_{tag}.csv")

    # Dead 
    pd.DataFrame((fD * [_.mean() for _ in model.D]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"cc100/Dx_{tag}.csv")

    # full simulation results
    pd.DataFrame(model.I ).to_csv(data/f"full_sims/It_{tag}.csv")
    pd.DataFrame(model.D ).to_csv(data/f"full_sims/Dt_{tag}.csv")
    pd.DataFrame(model.dT).to_csv(data/f"full_sims/dT_{tag}.csv")
    pd.DataFrame(model.dD).to_csv(data/f"full_sims/dD_{tag}.csv")

    if dVx_adm:
        # Administered vaccines
        pd.DataFrame(dVx_adm).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"cc100/Vx_adm_{tag}.csv")

        # Effective vaccines
        pd.DataFrame(dVx_eff).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"cc100/Vx_eff_{tag}.csv")

        # Immunizing vaccines
        pd.DataFrame(dVx_imm).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"cc100/Vx_imm_{tag}.csv")

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
for (district, seroprevalence, N_district, _, IFR_sero, _) in district_IFR.filter(items = district_codes.keys(), axis = 0).itertuples():
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

    print(district, I0, "extended IFR", ts.loc[district].dD.cumsum()[simulation_start]/T_scaled, "instantaneous IFR", dD0/(gamma * I0))

    # # run model forward with no vaccination
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
    param_tag = "novaccination"
    # ran_models[geo_tag + param_tag] = model

    t = 0
    while (model.Rt[-1].mean() > Rt_threshold) and (model.dT[-1].mean() > 0):
        model.parallel_forward_epi_step(num_sims = num_sims)
        print("::::", district, "no vax", t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean())
        t += 1
    save_results(model, data, [], [], [], geo_tag + param_tag)

    policy_deaths = np.sort(np.sum(model.dD, axis = 0))
    if param_tag in evaluated_deaths:
        evaluated_deaths[param_tag] += policy_deaths
    else:
        evaluated_deaths[param_tag]  = policy_deaths

    policy_YLL = np.sort((fD * np.sum(model.dD, axis = 0)).T @ YLLs[:, None], axis = None)
    if param_tag in evaluated_YLLs:
        evaluated_YLLs[param_tag] += policy_YLL
    else:
        evaluated_YLLs[param_tag]  = policy_YLL

    for (vax_pct_annual_goal, vax_effectiveness) in product(
        (0.5,),
        (0.70,)
    ):
        daily_rate = vax_pct_annual_goal/365
        daily_vax_doses = int(daily_rate * N_district)
        daily_net_doses = int(vax_effectiveness * daily_rate * N_district)

        policies = [
            RandomVaccineAssignment(daily_vax_doses, vax_effectiveness, split_by_age(S), IN_age_ratios), 
            PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), [6, 5, 4, 3, 2, 1, 0], "mortality"), 
            PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), [1, 2, 3, 4, 0, 5, 6], "contactrate")
        ]

        vaccination_policy: VaccinationPolicy
        for vaccination_policy in policies:
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
            param_tag = "_".join([
                vaccination_policy.name(),
                f"ve{int(100*vax_effectiveness)}",
                f"annualgoal{int(100 * vax_pct_annual_goal)}",
                f"Rt_threshold{Rt_threshold}"
            ])
            tag = geo_tag + param_tag
            # ran_models[geo_tag + param_tag] = model

            t = 0
            dVx_adm = [np.zeros(len(IN_age_structure))] # administered doses
            dVx_eff = [np.zeros(len(IN_age_structure))] # effective doses
            dVx_imm = [np.zeros(len(IN_age_structure))] # immunizing doses

            # run while Rt > threshold or until everyone is vaccinated 
            while (model.Rt[-1].mean() > Rt_threshold) and (not vaccination_policy.exhausted(model)):
                adm, eff, imm = vaccination_policy.distribute_doses(model, num_sims = num_sims)
                model.m       = vaccination_policy.get_mortality(list(TN_IFRs.values()))
                dVx_adm.append(adm)
                dVx_eff.append(eff)
                dVx_imm.append(imm)
                print("::::", district, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean(), model.S[-1].mean())
                t += 1

            save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag)
            
            policy_deaths = np.sum(model.dD, axis = 0)
            if param_tag in evaluated_deaths:
                evaluated_deaths[param_tag] += policy_deaths
            else:
                evaluated_deaths[param_tag]  = policy_deaths

            policy_YLL = (fD * np.sum(model.dD, axis = 0)).T @ YLLs[:, None]
            if param_tag in evaluated_YLLs:
                evaluated_YLLs[param_tag] += policy_YLL
            else:
                evaluated_YLLs[param_tag]  = policy_YLL

evaluated_death_percentiles = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_deaths.items()}
evaluated_YLL_percentiles   = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_YLLs.items()  }

# plot deaths
novax_death_percentiles     = {k: v for (k, v) in evaluated_death_percentiles.items() if "novaccination" in k}
random_death_percentiles    = {k: v for (k, v) in evaluated_death_percentiles.items() if "random"        in k}
contact_death_percentiles   = {k: v for (k, v) in evaluated_death_percentiles.items() if "contact"       in k}
mortality_death_percentiles = {k: v for (k, v) in evaluated_death_percentiles.items() if "mortality"     in k}

fig = plt.figure()
for (i, metric_percentiles) in enumerate([random_death_percentiles, contact_death_percentiles, mortality_death_percentiles]):
    plt.errorbar(
        x    = [0.2*(i-1) + dx for dx in range(len(metric_percentiles))],
        y    = [_[1] for _ in metric_percentiles.values()],
        yerr = list(zip(*[(_[1] - _[0], _[2] - _[1]) for _ in metric_percentiles.values()])),
        fmt = 'o',
        figure = fig,
        label = ["random assignment", "contact rate prioritized", "mortality prioritized"][i]
    )
(ticks, _) = plt.xticks()
tags = [_.split("_")[1:-2] for _ in metric_percentiles.keys()]
plt.xticks(ticks, labels = [""] + [v.replace("ve", "$v_e = $") + "\n" + p.replace("annualgoal", "$\phi = $") for (v, p) in tags], rotation = 0)
plt.legend(ncol = 3, title_fontsize = 18, fontsize = 16, framealpha = 1, handlelength = 1, bbox_to_anchor = (1.0090, 1), loc = "lower right")
plt.PlotDevice().ylabel("deaths\n")
plt.xticks(fontsize = "16")
plt.yticks(fontsize = "16")
plt.grid(False, which = "both", axis = "x")
ylims = plt.ylim() #(800, 1150)
# plt.vlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ymin = ylims[0], ymax = ylims[1])
# plt.xlim(left = -0.5, right = 5.5)
plt.ylim(*ylims)
plt.show()


# # plot YLLs
evaluated_YLL_percentiles   = {k: np.percentile(v, [0.05, 0.50, 0.95]) for (k, v) in evaluated_YLLs.items()}
novax_YLL_percentiles     = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "novaccination" in k}
random_YLL_percentiles    = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "random"        in k}
contact_YLL_percentiles   = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "contact"       in k}
mortality_YLL_percentiles = {k: v for (k, v) in evaluated_YLL_percentiles.items() if "mortality"     in k}

fig = plt.figure()
for (i, metric_percentiles) in enumerate([random_YLL_percentiles, contact_YLL_percentiles, mortality_YLL_percentiles]):
    plt.errorbar(
        x    = [0.2*(i-1) + dx for dx in range(len(metric_percentiles))],
        y    = [_[1] for _ in metric_percentiles.values()],
        yerr = list(zip(*[(_[1] - _[0], _[2] - _[1]) for _ in metric_percentiles.values()])),
        fmt = 'o',
        figure = fig,
        label = ["random assignment", "contact rate prioritized", "mortality prioritized"][i]
    )
(ticks, _) = plt.xticks()
tags = [_.split("_")[1:-2] for _ in metric_percentiles.keys()]
plt.xticks(ticks, labels = [""] + [v.replace("ve", "$v_e = $") + "\n" + p.replace("annualgoal", "$\phi = $") for (v, p) in tags], rotation = 0)
plt.legend(ncol = 3, title_fontsize = 18, fontsize = 16, framealpha = 1, handlelength = 1, bbox_to_anchor = (1.0090, 1), loc = "lower right")
plt.PlotDevice().ylabel("years of life lost\n")
plt.xticks(fontsize = "16")
plt.yticks(fontsize = "16")
plt.grid(False, which = "both", axis = "x")
ylims = plt.ylim() #(800, 1150)
plt.vlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ymin = ylims[0], ymax = ylims[1])
# plt.xlim(left = -0.5, right = 5.5)
plt.ylim(*ylims)
plt.show()

# fig = plt.figure()
# for (i, metric_percentiles) in enumerate([novax_death_percentiles, random_death_percentiles, contact_death_percentiles, mortality_death_percentiles]):
#     plt.errorbar(
#         x    = [0.2*(i-1) + dx for dx in range(len(metric_percentiles))],
#         y    = [_[1] for _ in metric_percentiles.values()],
#         yerr = list(zip(*[(_[1] - _[0], _[2] - _[1]) for _ in metric_percentiles.values()])),
#         fmt = 'o',
#         figure = fig,
#         label = ["no vaccination", "random assignment", "contact rate", "mortality"][i]
#     )
# (ticks, _) = plt.xticks()
# tags = [_.split("_")[1:-2] for _ in metric_percentiles.keys()]
# plt.xticks(ticks, labels = [""] + [v.replace("ve", "$v_e = $") + "\n" + p.replace("annualgoal", "$\phi = $") for (v, p) in tags], rotation = 0)
# plt.legend(ncol = 4, title_fontsize = 18, fontsize = 16, framealpha = 1, handlelength = 1)
# plt.PlotDevice().ylabel("deaths\n")
# plt.xticks(fontsize = "16")
# plt.yticks(fontsize = "16")
# plt.show()


# for district in ["Ariyalur", "Chengalpattu", "Chennai"]:
#     dT_conf_district = ts.loc[district].dT
#     dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
#     dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
#     T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
#     T = T_conf_smooth[date]
#     T_sero = (N_district * seroprevalence)
#     T_ratio = T_sero/T
    
#     model = ran_models[district]
#     plt.plot(range(len(dT)), dT)
#     plt.plot(range(len(dT), len(dT) + len(model.dT)), [_.mean() for _ in model.dT]) 
#     plt.title(district) 
#     plt.show() 

# for (_, model) in ran_models.items(): 
#     dD = ts.loc[model.name].dD
#     plt.scatter(range(len(dD)), dD)
#     plt.scatter(range(len(dD), len(dD) + len(model.dD)), [_.mean() for _ in model.dD]) 
#     plt.title(model.name) 
#     plt.show() 