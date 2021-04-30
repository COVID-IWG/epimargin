# import dask
import numpy as np
import pandas as pd
from epimargin.models import Age_SIRVD
from epimargin.utils import annually, normalize, percent, years
from studies.age_structure.commons import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("error")

num_sims         = 1000
simulation_range = 1 * years
phi_points       = [_ * percent * annually for _ in (25, 50, 100, 200)]
simulation_initial_conditions = pd.read_csv(data/f"all_india_coalesced_initial_conditions{simulation_start.strftime('%b%d')}.csv")\
    .drop(columns = ["Unnamed: 0"])\
    .set_index(["state", "district"])
rerun_states = ["Telangana", "Uttarakhand", "Jharkhand", "Arunachal Pradesh", "Nagaland", "Sikkim"] + coalesce_states
districts_to_run = simulation_initial_conditions
num_age_bins     = 7
seed             = 0

MORTALITY   = [6, 5, 4, 3, 2, 1, 0]
CONTACT     = [1, 2, 3, 4, 0, 5, 6]
CONSUMPTION = [4, 5, 6, 3, 2, 1, 0]

def save_metrics(tag, policy, dst = epi_dst):
    np.savez_compressed(dst/f"{tag}.npz", 
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
    return dV[:, sorted(range(len(prioritization)), key = prioritization.__getitem__)].clip(0, S)

def process(district_data):
    (
        (state, district), state_code, 
        sero_0, N_0, sero_1, N_1, sero_2, N_2, sero_3, N_3, sero_4, N_4, sero_5, N_5, sero_6, N_6, N_tot, 
        Rt, Rt_upper, Rt_lower, S0, I0, R0, D0, dT0, dD0, V0
    ) = district_data
    try: 
        S0 = int(S0)
    except ValueError as e:
        print (state, district, e)
        return 
    Sj0 = np.array([(1 - sj) * Nj for (sj, Nj) in zip([sero_0, sero_1, sero_2, sero_3, sero_4, sero_5, sero_6], [N_0, N_1, N_2, N_3, N_4, N_5, N_6])])
    # distribute historical doses assuming mortality prioritization
    Sj0 = prioritize(V0, Sj0.copy()[None, :], MORTALITY)[0]
    def get_model(seed = 0):
        model = Age_SIRVD(
            name        = state_code + "_" + district, 
            population  = N_tot - D0, 
            dT0         = (np.ones(num_sims) * dT0).astype(int), 
            Rt0         = Rt,
            S0          = np.tile( Sj0,        num_sims).reshape((num_sims, -1)),
            I0          = np.tile((fI * I0).T, num_sims).reshape((num_sims, -1)),
            R0          = np.tile((fR * R0).T, num_sims).reshape((num_sims, -1)),
            D0          = np.tile((fD * D0).T, num_sims).reshape((num_sims, -1)),
            mortality   = np.array(list(TN_IFRs.values())),
            infectious_period = infectious_period,
            random_seed = seed,
        )
        model.dD_total[0] = np.ones(num_sims) * dD0
        model.dT_total[0] = np.ones(num_sims) * dT0
        return model

    for phi in phi_points:
        num_doses = phi * (S0 + I0 + R0)
        sim_tag = f"{state_code}_{district}_phi{int(phi * 365 * 100)}_"
        random_model, mortality_model, contact_model, no_vax_model = [get_model(seed) for _ in range(4)]
        for t in range(simulation_range):
            if t <= 1/phi:
                dV_random    = num_doses * normalize(random_model.N[-1], axis = 1).clip(0)
                dV_mortality = prioritize(num_doses, mortality_model.N[-1], MORTALITY  ).clip(0) 
                dV_contact   = prioritize(num_doses, contact_model.N[-1],   CONTACT    ).clip(0) 
            else: 
                dV_random, dV_mortality, dV_contact = np.zeros((num_sims, 7)), np.zeros((num_sims, 7)), np.zeros((num_sims, 7))
            
            random_model   .parallel_forward_epi_step(dV_random,    num_sims = num_sims)
            mortality_model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)
            contact_model  .parallel_forward_epi_step(dV_contact,   num_sims = num_sims)
            no_vax_model   .parallel_forward_epi_step(dV = np.zeros((7, num_sims))[:, 0], num_sims = num_sims)

        if phi == phi_points[0]:
            save_metrics(sim_tag + "novax", no_vax_model   )
        save_metrics(sim_tag + "random",    random_model   )
        save_metrics(sim_tag + "mortality", mortality_model)
        save_metrics(sim_tag + "contact",   contact_model  )

if __name__ == "__main__":
    distribute = False
    if distribute:
        with dask.config.set({"scheduler.allowed-failures": 1}):
            client = dask.distributed.Client(n_workers = 1, threads_per_worker = 1)
            print(client.dashboard_link)
            with dask.distributed.get_task_stream(client) as ts:
                futures = []
                for district in districts_to_run.itertuples():
                    futures.append(client.submit(process, district, key = ":".join(district[0])))
            dask.distributed.progress(futures)
    else:
        failures = []
        for t in tqdm(districts_to_run.itertuples(), total = len(districts_to_run)):
            try: 
                process(t)
            except Exception as e:
                failures.append((e, t))
        for failure in failures:
            print(failure)
