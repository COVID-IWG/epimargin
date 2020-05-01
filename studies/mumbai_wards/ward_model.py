from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import etl
from adaptive.estimators import rollingOLS
from adaptive.model import Model, ModelUnit
from adaptive.plotting import gantt_chart, plot_curve
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks

seed  = 25
gamma = 0.2
Rv_Rm = 1.4836370631808469

def log_delta(ts):
    ld = pd.DataFrame(np.log(ts.cases.diff())).rename(columns = {"cases" : "logdelta"})
    ld["time"] = (ld.index - ld.index.min()).days
    return ld   

def model(wards, populations, cases, seed) -> Model:
    units = [
        ModelUnit(ward, populations[i], I0 = cases[ward].iloc[-1].cases) 
        for (i, ward) in enumerate(wards)
    ]
    return Model(units, random_seed=seed)

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    all_cases           = etl.load_case_data(data/"mumbai_wards_30Apr.csv")
    population_data     = etl.load_population_data(data/"ward_data_Mumbai_empt_slums.csv")["pop"]
    (wards, migrations) = etl.load_migration_data(data/"Ward_rly_matrix_Mumbai.csv")
    lockdown = np.zeros(migrations.shape)

    ward_cases = {ward: all_cases[all_cases.ward == ward] for ward in wards}
    tsw = {ward: log_delta(cases) for (ward, cases) in ward_cases.items()}
    grw = {ward: rollingOLS(ts) for (ward, ts) in tsw.items()}
    
    Rmw = {ward: np.mean(growth_rates.R) for (ward, growth_rates) in grw.items()}
    Rvw = {ward: Rv_Rm*Rm for (ward, Rm) in Rmw.items()}

    # 10 day lockdown 
    model_A = model(wards, population_data, ward_cases, seed)
    simulate_lockdown(model_A, 10*days, 190*days, Rmw, Rvw, lockdown, migrations)

    # 10 day + 4 week lockdown 
    model_B = model(wards, population_data, ward_cases, seed)
    simulate_lockdown(model_B, 10*days + 4*weeks, 190*days, Rmw, Rvw, lockdown, migrations)

    # 10 day lockdown + adaptive controls
    model_C = model(wards, population_data, ward_cases, seed).set_parameters(RR0 = Rmw)
    simulate_adaptive_control(model_C, 10*days, 190*days, lockdown, migrations, 
        {ward: Rv * gamma for (ward, Rv) in Rvw.items()},
        {ward: Rm * gamma for (ward, Rm) in Rmw.items()},
    )

    plot_curve([model_A, model_B, model_C], 
        ["Release on 05 May", "Release on 02 Jun", "Adaptive Controls from 05 May"], 
        "Mumbai", "Days Since 25 April", "Daily Infections", "Ward-Level Adaptive Controls")
    plt.show()

    gantt_chart(model_C.gantt, "Days Since 25 Apr", "Release Schedule by Ward")
    plt.show()
    
