from itertools import product

import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import SIR
from adaptive.policy import PrioritizedAssignment, RandomVaccineAssignment
from studies.age_structure.common_TN_data import *

sns.set(style = "whitegrid")

def save_results(model, data, dVx_adm, dVx_eff, dVx_imm, tag):
    print(":::: serializing results")
    
    # Susceptibles
    pd.DataFrame((IN_age_ratios [..., None] * [_.mean() for _ in model.S]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Sx_{tag}.csv")

    # Infectious 
    pd.DataFrame((sero_breakdown[..., None] * [_.mean() for _ in model.I]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Ix_{tag}.csv")

    # Recovered
    pd.DataFrame((IN_age_ratios [..., None] * [_.mean() for _ in model.R]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Rx_{tag}.csv")

    # Dead 
    pd.DataFrame((IN_age_ratios [..., None] * [_.mean() for _ in model.D]).astype(int)).T\
        .rename(columns = dict(enumerate(IN_age_structure.keys())))\
        .to_csv(data/f"Dx_{tag}.csv")

    if dVx_adm:
        # Administered vaccines
        pd.DataFrame(dVx_adm).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"Vx_adm_{tag}.csv")

        # Effective vaccines
        pd.DataFrame(dVx_eff).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"Vx_eff_{tag}.csv")

        # Immunizing vaccines
        pd.DataFrame(dVx_imm).cumsum()\
            .rename(columns = dict(enumerate(IN_age_structure.keys())))\
            .to_csv(data/f"Vx_imm_{tag}.csv")

# scaling
dT_conf = df[state].loc[:, "delta", "confirmed"] 
dT_conf_smooth = pd.Series(smooth(dT_conf), index = dT_conf.index)
T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
T = T_conf_smooth[date]
T_sero = (N * seropos)
T_ratio = T_sero/T

print(":: running simulations")
for (district, N_district) in district_populations.items():
    # grab timeseries 
    D, R = ts.loc[district][["dD", "dR"]].sum()

    dT_conf_district = ts.loc[district].dT
    dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
    dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
    T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
    S = N_district - T_scaled

    # run Rt estimation on scaled timeseries 
    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    geo_tag = f"{state}_{district}_"

    # # run model forward with no vaccination
    model = SIR(
        name        = state, 
        population  = N_district, 
        dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
        Rt0         = Rt[simulation_start] * N_district/(N_district - T_scaled),
        I0          = np.ones(num_sims) * (T_scaled - R - D), 
        R0          = np.ones(num_sims) * R, 
        D0          = np.ones(num_sims) * D,
        random_seed = 0
    )

    t = 0
    while (t < 5 * 365) and model.Rt[-1].mean() > Rt_threshold:
        model.parallel_forward_epi_step()
        print("::::", district, "no vax", t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean())
        t += 1
    save_results(model, data/"compartment_counts", [], [], [], geo_tag + "novaccination")

    for (vax_pct_annual_goal, vax_effectiveness) in product(
        (0.25, 0.50),
        (0.70, 1.00)
    ):
        daily_rate = vax_pct_annual_goal/365
        daily_vax_doses = int(daily_rate * N_district)
        daily_net_doses = int(vax_effectiveness * daily_rate * N_district)

        policies = [
            RandomVaccineAssignment(daily_vax_doses, vax_effectiveness, IN_age_ratios), 
            PrioritizedAssignment(daily_vax_doses, split_by_age(S), [6, 5, 4, 3, 2, 1, 0], "mortality"), 
            PrioritizedAssignment(daily_vax_doses, split_by_age(S), [1, 2, 3, 4, 0, 5, 6], "contactrate")
        ]

        for vaccination_policy in policies[:1]:
            model = SIR(
                name        = district, 
                population  = N_district, 
                dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
                Rt0         = Rt[simulation_start] * N_district/(N_district - T_scaled),
                I0          = np.ones(num_sims) * (T_scaled - R - D), 
                R0          = np.ones(num_sims) * R, 
                D0          = np.ones(num_sims) * D,
                random_seed = 0
            )

            t = 0
            dVx_adm = [np.zeros(len(IN_age_structure))] # administered doses
            dVx_eff = [np.zeros(len(IN_age_structure))] # effective doses
            dVx_imm = [np.zeros(len(IN_age_structure))] # immunizing doses

            # run vax rate forward 1 year, then until 75% of pop is recovered or vax
            while (t < 5 * 365) and model.Rt[-1].mean() > Rt_threshold:
                adm, eff, imm = vaccination_policy.distribute_doses(model)
                dVx_adm.append(adm)
                dVx_eff.append(eff)
                dVx_imm.append(imm)
                print("::::", district, vax_pct_annual_goal, vax_effectiveness, vaccination_policy.name(), t, np.mean(model.dT[-1]), np.std(model.dT[-1]), model.Rt[-1].mean())
                t += 1

            tag = geo_tag + "_".join([
                vaccination_policy.name(),
                f"ve{int(100*vax_effectiveness)}",
                f"annualgoal{int(100 * vax_pct_annual_goal)}",
                f"Rt_threshold{Rt_threshold}"
            ])
            save_results(model, data/"compartment_counts", dVx_adm, dVx_eff, dVx_imm, tag)
