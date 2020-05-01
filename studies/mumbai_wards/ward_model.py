from typing import Sequence

from adaptive.model import Model, ModelUnit
from adaptive.policy import simulate_lockdown, simulate_adaptive_control
from adaptive.utils import cwd
import etl

def units(wards, populations, timeseries) -> Sequence[ModelUnit]:
    pass 

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    timeseries = etl.load_case_data(data/"mumbai_wards_30Apr.csv")
    pop_data   = etl.load_population_data(data/"ward_data_Mumbai_empt_slums.csv")
    migrations = etl.load_migration_data(data/"Ward_rly_matrix_Mumbai.csv")

    simulate_lockdown()

    simulate_lockdown()

    simulate_adaptive_control()