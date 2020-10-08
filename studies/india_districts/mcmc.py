import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt
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

def linearized_SIR(infection_ts, gamma = 0.2, chains = 1, tune = 1000, draws = 1000, target_accept = 0.8):
    with pm.Model() as model:
        # lag new case counts
        diff = np.diff(infection_ts).clip(0)
        dT      = diff[2:]
        dT_lag1 = diff[1:-1]

        # set up random walk for every time point (adapted from Rt.live)
        step_size   = pm.HalfNormal('step_size', sigma=.03)
        theta_raw_0 = pm.Normal('theta_raw_init', 0.1, 0.1)
        theta_raw   = pm.Normal('theta_raw_steps', shape=len(dT)) * step_size
        theta = pm.Deterministic('theta', tt.concatenate([[theta_raw_0], theta_raw]).cumsum())

        # lambda ~ Gamma(alpha_l, beta_l)
        # bt     ~ Gamma(alpha_b, beta_b)
        # Rt     = 1 + ln(b_t)/gamma
        
        b_t      = pm.Gamma(alpha = alpha_b, beta = beta_b)
        lambda_t = pm.Deterministic("lambda_t", b_t * dT_lag1)
        dT_pred  = pm.Poisson(mu = lambda_t, observed = dT)
        return (model, pm.sample(chains = chains, tune = tune, draws = draws, target_accept = target_accept))

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
(model, trace) = linearized_SIR(ts.Hospitalized.values, 0.2)
pm.summary(trace)
pm.traceplot(trace)
plt.show()

def deconv(sigmal, filter):
    pass 

subtraction = pm.Model()
with subtraction:  
    delay = pm.Gamma("delay", mu = 200, sigma = 1)
    obs   = np.linspace(0, 100)
    orig  = pm.Deterministic("orig", obs - delay)
    trace = pm.sample(500)
    s = pm.summary(trace, hdi_prob = 0.95).drop("delay")
    s_mean = s["mean"].values
    s_CI_l = s["hdi_2.5%"].values
    s_CI_u = s["hdi_97.5%"].values
    
    plt.plot(obs, label = "original")
    plt.plot(s_mean, label = "subtracted mean")
    plt.fill_between(range(len(obs)), s_CI_l, s_CI_u, alpha = 0.4, label = "95% CI")
    plt.legend()
    plt.show()
