from pathlib import Path

import numpy as np
import pandas as pd
from numpy import diag, tile, vstack, eye

from adaptive.estimators import analytical_MPVS
from adaptive.smoothing import notched_smoothing
import adaptive.plots as plt

root = Path(__file__).parent
data = root/"data"

CI = 0.95
window = 10

def matrix_estimator(
        infection_ts: pd.DataFrame, 
        smoothing: Callable,
        alpha: float = 3.0,                # shape 
        beta:  float = 2.0,                # rate
        CI:    float = 0.95,               # confidence interval 
        infectious_period: int = 5*days,   # inf period = 1/gamma,
        variance_shift: float = 0.99,      # how much to scale variance parameters by when anomaly detected 
        totals: bool = False               # are these case totals or daily new cases?
    ):
    pass 


class AgeStructuredSIR():
    
    def forward_epi_step(num_sims):
        Rt = np.diag(S) @ self.Rt0 @ np.diag(1/N)
        Bt = np.exp(self.gamma @ (Rt - np.eye(self.num_age_bins)))
        dT = Poisson.rvs(mu = B @ dT0)

##############################
# load general purpose data

u = ()    # age specific infection probability 
# contact matrix from Laxminarayan 
C = np.array([

])
beta = u @ c # effective contact rates 

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

# age specific recovery rates 
g = ()
G = np.diag(g)

##############################
# Karnataka 
KA = pd.read_stata("data/ka_cases_deaths_time_newagecat.dta")
datemax = KA.date.max()

KA.agecat = KA.agecat.where(KA.agecat != 85, 75) # we don't have econ data for 85+ so combine 75+ and 85+ categories 
KA_agecases = KA.groupby(["agecat", "date"])["patientcode"]\
    .count().sort_index().rename("cases")\
    .unstack().fillna(0).stack()

KA_long = KA_agecases.unstack()

KA_ts = KA_agecases.sum(level = 1).sort_index()

## estimate overall rt
(
    dates, Rt_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates
) = analytical_MPVS(KA_ts, CI = CI, smoothing = notched_smoothing(window = window), totals = False)

plt.Rt(dates, Rt_pred, RR_CI_upper, RR_CI_lower, CI)
plt.show()
rt_KA_0 = np.mean(Rt_pred)

d0 = pd.concat([KA_long.T[0.0].rename("y"), KA_long.T.shift(-1).rename(lambda _: "x" + str(int(_)), axis = 1)], axis = 1).dropna()
params = sm.OLS.from_formula("y ~ x0 + x5 + x18 + x30 + x40 + x50 + x65 + x75 - 1", data = d0, hasconst = True).fit().params

B_matrix = []
for cat in [0.0, 5.0, 18.0, 30.0, 40.0, 50.0, 65.0, 75.0]:
    d = pd.concat([KA_long.T[cat].rename("y"), KA_long.T.shift(1).rename(lambda _: "x" + str(int(_)), axis = 1)], axis = 1).dropna()
    params = sm.OLS.from_formula("y ~ x0 + x5 + x18 + x30 + x40 + x50 + x65 + x75 - 1", data = d["July 1, 2020":]).fit().params
    B_matrix.append(params.values)
B = np.vstack(B_matrix)
Rt = np.eye(8) + 5 * np.log(B)
Rt.round(2)
## get age-structured Rt 

##############################
# other states 
