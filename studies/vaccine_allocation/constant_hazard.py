import numpy as np
import pandas as pd
from studies.vaccine_allocation.commons import *
from tqdm import tqdm
May15 = 30 # days since April 15

simulation_initial_conditions = pd.read_csv(data/f"all_india_coalesced_initial_conditions{simulation_start.strftime('%b%d')}.csv")\
    .drop(columns = ["Unnamed: 0"])\
    .set_index(["state", "district"])\
    .assign(
        frac_R  = lambda _:  _.R0         / _.N_tot, 
        frac_RV = lambda _: (_.R0 + _.V0) / _.N_tot,
        V0      = lambda _: _.V0.astype(int),
        D0      = lambda _: _.D0.astype(int),
        scaled_new_cases = lambda _: _.dT0.astype(int)
    )\
    [["Rt", "frac_R", "frac_RV", "V0", "scaled_new_cases"]]

def load_projections(state, district, t = May15):
    state_code = state_name_lookup[state]
    f = np.load(epi_dst / f'{state_code}_{district}_phi25_novax.npz')
    return [np.median(f["dD"], axis = 1).astype(int)[t], np.median(f["dD"], axis = 1).astype(int)[t]]

projections = [load_projections(*idx) for idx in tqdm(simulation_initial_conditions.index)]

# prioritization = simulation_initial_conditions\
#     .join(pd.DataFrame(projections, columns = ["projected_new_cases_may15", "projected_new_deaths_may15"], index = simulation_initial_conditions.index))

prioritization = pd.read_csv(data / "apr15_sero_prioritization.csv").set_index(["state", "district"])

crosswalk = pd.read_stata(Path.home() / "Dropbox/COVID Vaccination Policy/India/data/districts/all_crosswalk.dta")\
    .drop(columns = ["state", "district"])\
    .rename(columns = lambda s: s.replace("_api", ""))\
    .set_index(["state", "district"])\
    .sort_index()\
    .filter(like = "lgd", axis = 1)

crosswalk.loc[coalesce_states].reset_index()\
    .assign(
        district          = lambda _:_.state,
        lgd_district_id   = lambda _:_.lgd_state_id,
        lgd_district_name = lambda _:_.lgd_state_name
    ).drop_duplicates()

prioritization.join(pd.concat([
    crosswalk.drop(labels = coalesce_states),
    crosswalk.loc[coalesce_states].reset_index()\
        .assign(
            district          = lambda _:_.state,
            lgd_district_id   = lambda _:_.lgd_state_id,
            lgd_district_name = lambda _:_.lgd_state_name
        )\
        .drop_duplicates()\
        .set_index(["state", "district"])
]).sort_index())\
    .to_csv(data / "apr15_sero_prioritization_lgd.csv" )