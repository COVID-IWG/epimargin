import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

from adaptive.etl.covid19india import *
from adaptive.smoothing import convolution
from adaptive.utils import cwd

sns.set(palette="deep")

def plot(ts, window = 7):
    smooth = convolution(window = window)
    plt.bar(x = range(len(ts.values)), height = ts.values)
    plt.plot(smooth(ts.values), color = "white",  linewidth = 3)
    plt.plot(smooth(ts.values), color = "orange", linewidth = 1)
    plt.show()

def mcmc_gamma_prior(infection_ts, gamma, chains = 1, tune = 3000, draws = 1000, target_accept = 0.95):
    with pm.Model() as model:
        diff = np.diff(infection_ts).clip(0)
        delta_I      = diff[2:]
        delta_I_lag1 = diff[1:-1]

        # parameters for gamma distribution (branching parameter)
        # alpha = pm.Normal("alpha", mu = 3, sigma = 1, observed = delta_I.cumsum())
        # beta  = pm.Normal("beta",  mu = 2, sigma = 1, observed = delta_I_lag1.cumsum())
        alpha = delta_I.cumsum()
        beta  = delta_I_lag1.cumsum()

        # parameters for negative binomial (case prediction)
        # p = pm.Deterministic("p", beta/(delta_I_lag1 + beta))
        p = beta/(delta_I_lag1 + beta)
        p[np.isnan(p)] = 0

        # assume infection duration is distributed exponentially (cf. Abbott et al. @ CMMID/LHSTM)
        duration = pm.Exponential("duration", lam = gamma) 
        # # b(t) ~ Gamma(alpha, beta)
        branch_factor = pm.Gamma("branch_factor", alpha = alpha, beta = beta, testval = np.exp(0.9 * gamma))
        Rt = pm.Deterministic("Rt", 1 + duration * pm.math.log(branch_factor))

        cases = pm.NegativeBinomial("cases", mu = alpha, alpha = p, observed = infection_ts[3:])

        print("checking test point")
        print(model.check_test_point())

        print("sampling")
        return pm.sample(chains = chains, tune = tune, draws = draws, target_accept = target_accept)

root = cwd()
data = root/"data"
if not data.exists():
    data.mkdir()

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in (3, 4, 5, 6, 7, 8)]
}

print("setting up data")
df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)

ts = get_time_series(df)

print("running estimator")
trace = mcmc_gamma_prior(ts.Hospitalized.values, 0.2)
