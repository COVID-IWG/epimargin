from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.models import Age_SIRVD
from adaptive.utils import annually, normalize, percent, years
from scipy.stats import multinomial
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from studies.age_structure.wtp import *
from tqdm.std import tqdm

sns.set(style = "whitegrid")

num_sims         = 500
simulation_range = 1 * years
phi_points       = [_ * percent * annually for _ in (25, 50, 100, 200)]
simulation_initial_conditions = pd.read_csv(data/"simulation_initial_conditions.csv").set_index("district")
districts_to_run = simulation_initial_conditions
# districts_to_run = district_IFR.filter(items = ["Chennai", "Cuddalore", "Salem"], axis = 0)
num_age_bins     = 7
seed             = 0

dst = data/"sim_metrics_100"
dst.mkdir(exist_ok = True)

def save_metrics(policy, tag):
    np.savez(dst/f"{tag}.npz", 
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
    progress = tqdm(total = (simulation_range + 4) * len(districts_to_run) * len(phi_points), leave = False)
    for (district, sero_0, N_0, sero_1, N_1, sero_2, N_2, sero_3, N_3, sero_4, N_4, sero_5, N_5, sero_6, N_6, N_tot, Rt, S0, I0, R0, D0, dT0, dD0) in districts_to_run.itertuples():
        Sj0 = np.array([(1 - sj) * Nj for (sj, Nj) in zip([sero_0, sero_1, sero_2, sero_3, sero_4, sero_5, sero_6], [N_0, N_1, N_2, N_3, N_4, N_5, N_6])])
        def get_model(seed = 104):
            model = Age_SIRVD(
                name        = district, 
                population  = N_tot - D0, 
                dT0         = np.ones(num_sims) * (dT0).astype(int), 
                Rt0         = Rt,
                S0          = np.tile(Sj0,         num_sims).reshape((num_sims, -1)),
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

            # D_diff_contact[int(phi * 100 * 365)]   = np.squeeze(contact_model.D)   - np.squeeze(no_vax_model.D)
            # D_diff_random[int(phi * 100 * 365)]    = np.squeeze(random_model.D)    - np.squeeze(no_vax_model.D)
            # D_diff_mortality[int(phi * 100 * 365)] = np.squeeze(mortality_model.D) - np.squeeze(no_vax_model.D)

            # {k: (v.sum(axis = 2)[-1] > 0).sum() for (k, v) in D_diff_contact.items()}
            # {k: (v.sum(axis = 2)[-1] > 0).sum() for (k, v) in D_diff_random.items()}
            # {k: (v.sum(axis = 2)[-1] > 0).sum() for (k, v) in D_diff_mortality.items()}

            # for (model, modelname) in zip((random_model, contact_model, mortality_model, no_vax_model), ("random", "contact", "mortality", "no vax")):
            #     print(int(phi * 100 * 365), modelname, [op(np.sum(model.D, axis = 2)[-1]) for op in [np.min, np.mean, np.max]])
