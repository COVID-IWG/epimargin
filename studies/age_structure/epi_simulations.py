import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import Age_SIRVD
from adaptive.utils import annually, normalize, percent, years
from scipy.stats import multinomial
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from studies.age_structure.wtp import *
from tqdm.std import tqdm

sns.set(style = "whitegrid")

num_sims         = 1000
simulation_range = 1 * years
phi_points       = [_ * percent * annually for _ in (25, 50, 100)]
districts_to_run = district_IFR.filter(items = sorted(set(district_codes.keys()) - set(["Perambalur"])), axis = 0)
# districts_to_run = district_IFR.filter(items = ["Chennai", "Cuddalore", "Salem"], axis = 0)
num_age_bins     = 7
seed             = 0

def save_metrics(policy, tag):
    np.savez(
        data/f"sim_metrics/{tag}.npz", 
        dT = policy.dT_total,
        dD = policy.dD_total,
        pi = policy.pi,
        q0 = policy.q0,
        q1 = policy.q1, 
        Dj = policy.D
    )

def prioritize(num_doses, S, prioritization):
    Sp = S[:, prioritization]
    dV = np.where(Sp.cumsum(axis = 1) <= num_doses, Sp, 0)
    dV[np.arange(len(dV)), (Sp.cumsum(axis = 1) > dV.cumsum(axis = 1)).argmax(axis = 1)] = num_doses - dV.sum(axis = 1)
    return dV[:, [i for (i, _) in sorted(enumerate(prioritization), key = lambda t: t[1])]]

if __name__ == "__main__":
    progress = tqdm(total = (simulation_range + 4) * len(districts_to_run) * len(phi_points))
    for (district, seroprevalence, N_district, _, IFR_sero, _) in districts_to_run.itertuples():
        dT_conf_district = ts.loc[district].dT
        dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
        dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
        T_conf_smooth = dT_conf_district_smooth.cumsum().astype(int)
        T = T_conf_smooth[date]
        T_sero = (N_district * seroprevalence)
        T_ratio = T_sero/T
        D0, R0 = ts.loc[district][["dD", "dR"]].sum()
        R0 *= T_ratio

        T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
        S0 = N_district - T_scaled
        dD0 = ts.loc[district].dD.loc[simulation_start]
        dT0 = ts.loc[district].dT.loc[simulation_start] * T_ratio
        I0 = max(0, (T_scaled - R0 - D0))

        (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
        Rt = dict(zip(Rt_dates, Rt_est))

        def get_model(seed = 104):
            model = Age_SIRVD(
                name        = district, 
                population  = N_district - D0, 
                dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
                Rt0         = Rt[simulation_start],
                S0          = np.tile((fS * S0).T, num_sims).reshape((num_sims, -1)),
                I0          = np.tile((fI * I0).T, num_sims).reshape((num_sims, -1)),
                R0          = np.tile((fR * R0).T, num_sims).reshape((num_sims, -1)),
                D0          = np.tile((fD * D0).T, num_sims).reshape((num_sims, -1)),
                mortality   = np.array(list(TN_IFRs.values())), #(ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
                random_seed = seed
            )
            model.dD_total[0] = np.ones(num_sims) * dD0
            model.dT_total[0] = np.ones(num_sims) * dT0
            return model

        for phi in phi_points:
            progress.set_description(f"{district:15s}(φ = {str(int(365 * 100 * phi)):>3s}%)")
            num_doses = phi * (S0 + I0 + R0)
            random_model, mortality_model, contact_model, no_vax_model = [get_model(seed) for _ in range(4)]
            for t in range(simulation_range):
                if t <= 1/phi:
                    dV_random    = multinomial.rvs(num_doses, normalize(random_model.N[-1], axis = 1)[0]).reshape((-1, 7))
                    dV_mortality = prioritize(num_doses, mortality_model.S[-1], [6, 5, 4, 3, 2, 1, 0]) 
                    dV_contact   = prioritize(num_doses, contact_model.S[-1],   [1, 2, 3, 4, 0, 5, 6]) 
                else: 
                    dV_random, dV_mortality, dV_contact = np.zeros((num_sims, 7)), np.zeros((num_sims, 7)), np.zeros((num_sims, 7))
                
                random_model.parallel_forward_epi_step(dV_random, num_sims = num_sims)
                mortality_model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)
                contact_model.parallel_forward_epi_step(dV_contact, num_sims = num_sims)
                no_vax_model.parallel_forward_epi_step(dV = np.zeros((7, num_sims))[:, 0], num_sims = num_sims)
                progress.update(1)

            save_metrics(no_vax_model,    f"{state}_{district}_phi{int(phi * 365 * 100)}_novax")
            progress.update(1)
            save_metrics(random_model,    f"{state}_{district}_phi{int(phi * 365 * 100)}_random")
            progress.update(1)
            save_metrics(mortality_model, f"{state}_{district}_phi{int(phi * 365 * 100)}_mortality")
            progress.update(1)
            save_metrics(contact_model,   f"{state}_{district}_phi{int(phi * 365 * 100)}_contact")
            progress.update(1)
