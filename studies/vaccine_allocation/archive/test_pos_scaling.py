import epimargin.plots as plt
import flat_table
import numpy as np
import pandas as pd
from epimargin.estimators import analytical_MPVS
from epimargin.etl.covid19india import state_code_lookup
from epimargin.smoothing import notched_smoothing
from epimargin.utils import days, setup
from statsmodels.api import OLS
from statsmodels.iolib.summary2 import summary_col
from scipy.signal import convolve, deconvolve

from scipy.stats import expon as Exponential, gamma as Gamma


data, _ = setup()

# load population data 
india_pop = pd.read_csv(data/"india_pop.csv", names = ["state", "population"], index_col = "state").to_dict()["population"]
india_pop["India"]       = sum(india_pop.values())
india_pop["Odisha"]      = india_pop["Orissa"]
india_pop["Puducherry"]  = india_pop["Pondicherry"]
india_pop["Uttarakhand"] = india_pop["Uttaranchal"]

# load case data 
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)


state = "KA"
# state = "TN"
# state = "TT"

# assemble observation matrix
obs = pd.concat([df[state][:, "delta", "tested"], df[state][:, "delta", "confirmed"], df[state][:, "delta", "recovered"], df[state][:, "delta", "deceased"]], axis = 1)
obs.columns = ["tested", "confirmed", "recovered", "deceased"]
obs = obs.reindex(pd.date_range(obs.index.min(), obs.index.max()), fill_value = 0)
obs["month"] = obs.index.month
obs["tests_per_mill"] = obs["tested"]/(india_pop[state_code_lookup[state]]/1e6)

# define regression formula as function of taylor approximation order for rate-control function
def scaling(order: int) -> str: 
    powers = " + ".join(f"np.power(tests_per_mill, {i + 1})" for i in range(order)) # test rate exponentiation terms
    return f"confirmed ~ -1 + tested + C(month)*({powers})" # no intercept, regress on tests, interact month indicator with powers 

# select order by minimizing AIC where coefficient on number of tests > 0
models = [OLS.from_formula(scaling(order), data = obs).fit() for order in range(1, 10)]
(model_idx, selected_model) = min(((i, each) for (i, each) in enumerate(models) if each.params["tested"] > 0), key = lambda _: _[1].aic)
print("  i aic     r2   beta")
for (i, model) in enumerate(models):
    print("*" if i == model_idx else " ", i+1, model.aic.round(2), model.rsquared.round(2), model.params["tested"].round(2))
scale_factor = selected_model.params["tested"]

plt.plot(0.2093       * df[state][:, "delta", "tested"],    label = "national test-scaled")
plt.plot(scale_factor * df[state][:, "delta", "tested"],    label = "state test-scaled")
plt.plot(               df[state][:, "delta", "confirmed"], label = "confirmed")
plt.legend()
plt.PlotDevice().title(f"\n{state} / case scaling comparison").xlabel("\ndate").ylabel("cases\n")
plt.show()

# I vs D estimators 
gamma  = 0.2
window = 7 * days
CI     = 0.95
smooth = notched_smoothing(window)

(dates_I, Rt_I, Rtu_I, Rtl_I, *_) = analytical_MPVS(df[state][:, "delta", "confirmed"], CI = CI, smoothing = smooth, totals = False)
(dates_D, Rt_D, Rtu_D, Rtl_D, *_) = analytical_MPVS(df[state][:, "delta", "deceased" ], CI = CI, smoothing = smooth, totals = False)

plt.Rt(dates_I, Rt_I, Rtu_I, Rtl_I, CI)\
    .title(f"{state} - $R_t(I)$ estimator")
plt.figure()
plt.Rt(dates_D, Rt_D, Rtu_D, Rtl_D, CI)\
    .title(f"{state} - $R_t(D)$ estimator")

plt.show()

KA_dD = df["KA"][:, "delta", "deceased"]
KA_D  = KA_dD.cumsum()
L = 14
dist = Exponential(scale = 1/L)
pmf  = dist.pdf(np.linspace(dist.ppf(0.005), dist.ppf(1 - 0.005)))
pmf /= pmf.sum()

D_deconv, _ = deconvolve(KA_D, pmf)
D_deconv *= 1/0.02
plt.plot(KA_D.values.clip(0.1), label = "deaths")
plt.plot(D_deconv.clip(0.1),     label = "deconv")
plt.legend()
# plt.semilogy()
plt.show()
