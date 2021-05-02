from pathlib import Path

import epimargin.plots as plt
import flat_table
import numpy as np
import pandas as pd
import seaborn as sns
from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import state_code_lookup
from epimargin.models import SIR
from epimargin.smoothing import notched_smoothing

sns.set(style = "whitegrid")

# root = Path(__file__).parent
# data = root/"data"
data = Path("./data").resolve()

CI = 0.95
window = 14
gamma = 0.2
infectious_period = 5
smooth = notched_smoothing(window)
num_sims = 50000

# load admin data on population
IN_age_structure = { # WPP2019_POP_F01_1_POPULATION_BY_AGE_BOTH_SEXES
    0:  116_880,
    5:  117_982 + 126_156 + 126_046,
    18: 122_505 + 117_397, 
    30: 112_176 + 103_460,
    40: 90_220 + 79_440,
    50: 68_876 + 59_256 + 48_891,
    65: 38_260 + 24_091,
    75: 15_084 + 8_489 + 3_531 +  993 +  223 +  48,
}
# normalize
age_structure_norm = sum(IN_age_structure.values())
IN_age_ratios = np.array([v/age_structure_norm for (k, v) in IN_age_structure.items()])
split_by_age = lambda v: (v * IN_age_ratios).astype(int)

# from Karnataka
COVID_age_ratios = np.array([0.01618736, 0.07107746, 0.23314877, 0.22946212, 0.18180406, 0.1882451 , 0.05852026, 0.02155489])

india_pop = pd.read_csv(data/"india_pop.csv", names = ["state", "population"], index_col = "state").to_dict()["population"]
india_pop["Odisha"]      = india_pop["Orissa"]
india_pop["Puducherry"]  = india_pop["Pondicherry"]
india_pop["Uttarakhand"] = india_pop["Uttaranchal"]

# load covid19 india data 
download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)

# load Rt data 
# Rt = pd.read_csv("data/Rt_timeseries_india.csv") 
# date = "2020-12-24"
# for state in set(df.loc[date, :, :].columns) - {"TT", "LA", "SK", "NL"}: 
# for state in ["TN"]:
#     N = india_pop[state_code_lookup[state].replace("&", "and")]
#     T = df[state].loc[date, "total", "confirmed"]
#     R = df[state].loc[date, "total", "recovered"]
#     D = df[state].loc[date, "total", "deceased"]
#     model = SIR(
#         name        = state, 
#         population  = N - D, 
#         dT0         = np.ones(num_sims) * df[state].loc[date, "delta", "confirmed"], 
#         Rt0         = Rt[(Rt.state == state) & (Rt.date == date)].Rt.iloc[0], 
#         I0          = np.ones(num_sims) * (T - R - D), 
#         R0          = np.ones(num_sims) * R, 
#         D0          = np.ones(num_sims) * D,
#         random_seed = 0
#     )
#     i = 0
#     while np.mean(model.dT[-1]) > 0:
#         model.parallel_forward_epi_step(num_sims = num_sims)
#         i += 1
#         print(state, i, np.mean(model.dT[-1]), np.std(model.dT[-1]))
#     dT  = np.array([_.mean().astype(int) for _ in model.dT])
#     dTx = (dT * COVID_age_ratios[..., None]).astype(int)
#     Tx  = (T * COVID_age_ratios).astype(int)[..., None] + dTx.cumsum(axis = 1)
#     Nx  = split_by_age(N)
#     lambda_x = dT/(Nx[..., None] - Tx)
#     pd.DataFrame(lambda_x).to_csv(data/f"{state}_age_hazards.csv")
#     pd.DataFrame(model.dT).T.to_csv(data/f"{state}_sims.csv")

######################
# sero scaling

TN_sero_breakdown = np.array([0.311, 0.311, 0.311, 0.320, 0.333, 0.320, 0.272, 0.253]) # from TN sero, assume 0-18 sero = 18-30 sero
TN_pop = india_pop["Tamil Nadu"]
TN_seropos = split_by_age(TN_pop) @ TN_sero_breakdown/TN_pop

#KA_seropos = 0.467 # statewide from KA private survey
KA_seropos = 0.273 # statewide from KA govt survey

scaled_Rt = { 
    "TN": 0.9271785447646147,
    # "KA": 1.1929944867195017
    "KA": 0.9636985404892338
}

simulation_start = pd.Timestamp("Jan 1, 2021")
smoothing = notched_smoothing(14)
num_sims = 10000

for (state, date, seropos, sero_breakdown) in (
    ("TN", "October 23, 2020", TN_seropos, TN_sero_breakdown), 
    #("KA", "2020-07-22", KA_seropos, IN_age_ratios)
    # ("KA", "2020-09-16", KA_seropos, IN_age_ratios),
    ):

    N = india_pop[state_code_lookup[state].replace("&", "and")]
    
    # scaling
    dT_conf = df[state].loc[:, "delta", "confirmed"] 
    dT_conf_smooth = pd.Series(smoothing(dT_conf), index = dT_conf.index)
    T_conf_smooth = dT_conf_smooth.cumsum().astype(int)
    T = T_conf_smooth[date]
    T_sero = (N * seropos)
    T_ratio = T_sero/T

    # grab time series 
    R = df[state].loc[simulation_start, "total", "recovered"]
    D = df[state].loc[simulation_start, "total", "deceased"]

    # run Rt estimation on scaled timeseries 
    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    model = SIR(
        name        = state, 
        population  = N, 
        dT0         = np.ones(num_sims) * (dT_conf_smooth[simulation_start] * T_ratio).astype(int), 
        Rt0         = Rt[simulation_start] * N/(N - T_sero),
        I0          = np.ones(num_sims) * (T_sero - R - D), 
        R0          = np.ones(num_sims) * R, 
        D0          = np.ones(num_sims) * D,
        random_seed = 0
    )
    i = 0

    print(Rt[simulation_start], Rt[simulation_start] * N/(N - T_ratio * T_conf_smooth[simulation_start]))

    while np.mean(model.dT[-1]) > 0:
        model.parallel_forward_epi_step(num_sims = num_sims)
        i += 1
        print(state, i, np.mean(model.dT[-1]), np.std(model.dT[-1]))

    # plot simulation
    plt.scatter(dT_conf["April 1, 2020":simulation_start].index, dT_conf["April 1, 2020":simulation_start].values*T_ratio,  label = "seroprevalence-scaled cases (pre-simulation)", color = "black", s = 5)
    # plt.scatter(dT_conf[simulation_start:].index, dT_conf[simulation_start:].values*T_ratio, color = "grey", label = "seroprevalence-scaled cases (post-simulation)", s = 5)
    # t = pd.Timestamp(date)
    dates = pd.date_range(simulation_start, simulation_start + pd.Timedelta(len(model.dT) - 1, "days"))
    # dates = pd.date_range(t, pd.Timestamp("April 1, 2021"))
    n = len(dates)
    plt.plot(dates, np.array([_.mean().astype(int) for _ in model.dT][:n]), label = "mean simulated daily cases", color = "rebeccapurple")
    plt.fill_between(dates, [_.min().astype(int) for _ in model.dT][:n], [_.max().astype(int) for _ in model.dT][:n], label = "simulation range", alpha = 0.3, color = "rebeccapurple")
    plt.vlines(pd.Timestamp(date), 1, 1e6, linestyles = "dashed", label = "date of seroprevalence study")
    plt.legend(handlelength = 1, framealpha = 1)
    plt.semilogy()
    plt.xlim(pd.Timestamp("April 1, 2020"), dates[-1])
    plt.ylim(1, 1e6)
    plt.PlotDevice().xlabel("\ndate").ylabel("new daily cases\n").annotate("Daily Cases: Scaled Data & Simulation - Tamil Nadu, no vaccination")
    plt.show()

    # calculate hazards
    dT  = np.array([_.mean().astype(int) for _ in model.dT])
    S   = np.array([_.mean().astype(int) for _ in model.S])
    dTx = (dT * sero_breakdown[..., None]).astype(int)
    Sx  = (S  * COVID_age_ratios[..., None]).astype(int)
    lambda_x = dTx/Sx
    Pr_covid_t     = np.zeros(lambda_x.shape)
    Pr_covid_pre_t = np.zeros(lambda_x.shape)
    Pr_covid_t[:, 0]     = lambda_x[:, 0]
    Pr_covid_pre_t[:, 0] = lambda_x[:, 0]
    for t in range(1, len(lambda_x[0, :])):
        Pr_covid_t[:, t] = lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])
        Pr_covid_pre_t[:, t] = Pr_covid_pre_t[:, t-1] + lambda_x[:, t] * (1 - Pr_covid_pre_t[:, t-1])
    
    plt.figure()
    for _ in range(8):
        plt.plot(Pr_covid_pre_t[_, :], label = f"agecat:{_}")
    plt.title(f"{state}: Pr(covid before t) - Tamil Nadu, no vaccination")
    plt.legend()
    plt.show()

    plt.figure()
    for _ in range(8):
        plt.plot(Pr_covid_t[_, :], label = f"agecat:{_}")
    plt.title(f"{state}: Pr(covid at t) - Tamil Nadu, no vaccination")
    plt.legend()
    plt.show()
    # Tx  = (T * sero_breakdown).astype(int)[..., None] + dTx.cumsum(axis = 1)
    # Nx  = split_by_age(N)
    # lambda_x = dTx/(Nx[..., None] - Tx)
    pd.DataFrame(lambda_x).to_csv(data/f"{state}_scaled_age_hazards_Jan_1.csv")
    pd.DataFrame(model.dT).T.to_csv(data/f"{state}_scaled_sims_Jan_1.csv")
    pd.DataFrame(Pr_covid_pre_t).to_csv(data/f"{state}_pr_cov_pre_t_Jan_1.csv")
    pd.DataFrame(Pr_covid_t).to_csv(data/f"{state}_pr_cov_at_t_Jan_1.csv")
