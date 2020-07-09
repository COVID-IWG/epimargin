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



def mcmc_gamma_prior(infection_ts, chains = 1, tune = 3000, draws = 1000, target_accept = 0.95):
    with pm.Model():
        # priors on shape and rate for negative binomial 
        alpha = pm.Normal('alpha', mu = 3, sigma = 1) 
        beta  = pm.Normal('beta',  mu = 2, sigma = 1)

        # assume infection duration is distributed exponentially (cf. CMMID @ LSTHM)
        duration = pm.Exponential(lam = gamma)

        # b(t) ~ Gamma(alpha, beta)
        branch_factor = pm.Gamma(alpha = alpha, beta = beta)

        Rt = pm.Deterministic("Rt", 1 + duration * pm.math.log(branch_factor))

        cases = pm.NegativeBinomial()

        return pm.sample(chains = chains, tune = tune, draws = draws, target_accept = target_accept)

 

root = cwd()
data = root/"data"
if not data.exists():
    data.mkdir()

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in (3, 4, 5, 6, 7)]
}

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)

ts = get_time_series(df)
