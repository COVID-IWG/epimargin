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

num_sims = 5000

def save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag):
    print(":::: serializing results")

    if "novaccination" in tag:
        pd.DataFrame(model.dT).to_csv(data/f"full_sims/dT_{tag}.csv")
        pd.DataFrame(model.dD).to_csv(data/f"full_sims/dD_{tag}.csv")

    # Dead 
    pd.DataFrame((fD * [_.mean() for _ in model.D]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"correct_sims/Dx_{tag}.csv")

    # # Susceptibles
    # pd.DataFrame((fS * [_.mean() for _ in model.S]).astype(int)).T\
    #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #     .to_csv(data/f"cc100/Sx_{tag}.csv")

    # # Infectious 
    # pd.DataFrame((fI * [_.mean() for _ in model.I]).astype(int)).T\
    #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #     .to_csv(data/f"cc100/Ix_{tag}.csv")

    # # Recovered
    # pd.DataFrame((fR * [_.mean() for _ in model.R]).astype(int)).T\
    #     .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #     .to_csv(data/f"cc100/Rx_{tag}.csv")

    # full simulation results
    # pd.DataFrame(model.I ).to_csv(data/f"full_sims/I_{tag}.csv")
    # pd.DataFrame(model.D ).to_csv(data/f"full_sims/D_{tag}.csv")

    # if dVx_adm:
    #     # Administered vaccines
    #     pd.DataFrame(dVx_adm).cumsum()\
    #         .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #         .to_csv(data/f"cc100/Vx_adm_{tag}.csv")

    #     # Effective vaccines
    #     pd.DataFrame(dVx_eff).cumsum()\
    #         .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #         .to_csv(data/f"cc100/Vx_eff_{tag}.csv")

    #     # Immunizing vaccines
    #     pd.DataFrame(dVx_imm).cumsum()\
    #         .rename(columns = dict(enumerate(IN_age_structure.keys())))\
    #         .to_csv(data/f"cc100/Vx_imm_{tag}.csv")

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

# for (district, seroprevalence, N_district, _, IFR_sero, _) in district_IFR.filter(items = district_codes.keys(), axis = 0).itertuples():
for (district, seroprevalence, N_district, _, IFR_sero, _) in district_IFR.filter(items = ["Chennai"], axis = 0).itertuples():
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
        (0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0),
        (0.5, 0.7, 1.0,)
    ):
        daily_rate = vax_pct_annual_goal/365
        daily_vax_doses = int(daily_rate * N_district)
        daily_net_doses = int(vax_effectiveness * daily_rate * N_district)

        policies = [
            RandomVaccineAssignment(
                daily_vax_doses, 
                vax_effectiveness, 
                np.tile(split_by_age(S), (num_sims, 1)), 
                np.tile(np.squeeze(fI * I0), (num_sims, 1)),
                IN_age_ratios, 
                np.array(list(TN_IFRs.values()))) 
            # PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), np.squeeze(fI * I0), [6, 5, 4, 3, 2, 1, 0], np.array(list(TN_IFRs.values())), "mortality"), 
            # PrioritizedAssignment(  daily_vax_doses, vax_effectiveness, split_by_age(S), np.squeeze(fI * I0), [1, 2, 3, 4, 0, 5, 6], np.array(list(TN_IFRs.values())), "contactrate")
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

            while t < 5 * 365:
                adm, eff, imm = vaccination_policy.distribute_doses(model, num_sims = num_sims)
                dVx_adm.append(adm)
                dVx_eff.append(eff)
                dVx_imm.append(imm)
                model.m = vaccination_policy.update_mortality()
                print("::::", district, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean(), model.S[-1].mean())
                t += 1
                1/0

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


plt.hist(
    [
        # evaluated_deaths["novaccination"], 
        evaluated_deaths["randomassignment_ve100_annualgoal0_Rt_threshold0.2"], 
        evaluated_deaths["randomassignment_ve100_annualgoal1_Rt_threshold0.2"],
        evaluated_deaths["randomassignment_ve100_annualgoal100_Rt_threshold0.2"],
        evaluated_deaths["randomassignment_ve100_annualgoal1000_Rt_threshold0.2"],
    ], 
    bins = list(range(0, 600, 30)),
    label = [
        # "no vax", 
        "random, phi = 0%", "random, phi = 1%", "random, phi = 100%", "random, phi = 1000%"]
)
plt.PlotDevice().title("\ndistribution of deaths")
plt.legend()
plt.xlim(left = 0, right = 600)
plt.show()

evaluated_death_percentiles = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_deaths.items()}
evaluated_death_percentiles.pop("novaccination")
labels = [key.split("_")[2].replace("annualgoal", "") for key in evaluated_death_percentiles.keys()]
fig = plt.figure()
for (i, (key, (lo, md, hi))) in enumerate(evaluated_death_percentiles.items()):
    plt.errorbar(
        x = [i], y = [md], yerr = [[md - lo], [hi - md]],
        figure = fig, fmt = "o", label = "$\phi =$ " + labels[i]
    )
plt.xticks(list(range(len(evaluated_death_percentiles))), labels)
plt.PlotDevice().title("death percentiles (5, 50, 95)")
plt.legend()
plt.show()

evaluated_death_means = {k:[v.min(), v.mean(), v.max()] for (k, v) in evaluated_deaths.items()}   
evaluated_death_means.pop("novaccination")
labels = [key.split("_")[2].replace("annualgoal", "") for key in evaluated_death_means.keys()]
fig = plt.figure()
for (i, (key, (lo, md, hi))) in enumerate(evaluated_death_means.items()):
    plt.errorbar(
        x = [i], y = [md], yerr = [[md - lo], [hi - md]],
        figure = fig, fmt = "o", label = "$\phi =$ " + labels[i]
    )
plt.xticks(list(range(len(evaluated_death_means))), labels)
plt.PlotDevice().title("death ranges (min, avg, max)")
plt.show()

evaluated_YLL_percentiles = {k: np.percentile(v, [5, 50, 95]) for (k, v) in evaluated_YLLs.items()}
evaluated_YLL_percentiles.pop("novaccination")
labels = [key.split("_")[2].replace("annualgoal", "") for key in evaluated_YLL_percentiles.keys()]
fig = plt.figure()
for (i, (key, (lo, md, hi))) in enumerate(evaluated_YLL_percentiles.items()):
    plt.errorbar(
        x = [i], y = [md], yerr = [[md - lo], [hi - md]],
        figure = fig, fmt = "o", label = "$\phi =$ " + labels[i]
    )
plt.xticks(list(range(len(evaluated_YLL_percentiles))), labels)
plt.PlotDevice().title("YLL percentiles (5, 50, 95)")
plt.legend()
plt.show()

evaluated_YLL_means = {k:[v.min(), v.mean(), v.max()] for (k, v) in evaluated_YLLs.items()}   
evaluated_YLL_means.pop("novaccination")
labels = [key.split("_")[2].replace("annualgoal", "") for key in evaluated_YLL_means.keys()]
fig = plt.figure()
for (i, (key, (lo, md, hi))) in enumerate(evaluated_YLL_means.items()):
    plt.errorbar(
        x = [i], y = [md], yerr = [[md - lo], [hi - md]],
        figure = fig, fmt = "o", label = "$\phi =$ " + labels[i]
    )
plt.xticks(list(range(len(evaluated_YLL_means))), labels)
plt.PlotDevice().title("YLL ranges (min, avg, max)")
plt.show()