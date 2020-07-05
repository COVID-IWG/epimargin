import pymc3 as pm 

import seaborn as sns 
sns.set(palette="deep")

def plot(ts, window = 7, local_window = 3):
    plt.bar(x = range(len(ts.values)), height = ts.values)
    plt.plot(box_filter(ts.values, window, local_window), color = "white",  linewidth = 3)
    plt.plot(box_filter(ts.values, window, local_window), color = "orange", linewidth = 1)
    plt.show()

plot(ts.delta)
plot(dT)


with pm.Model():
    # priors on shape and rate 
    alpha = pm.Normal('alpha', mu = 3, sigma = 10)
    beta  = pm.Normal('beta',  mu = 2, sigma = 10)

    # assume infection duration is distributed exponentially (cf. CMMID @ LSTHM)
    duration = pm.Exponential(lam = gamma)

    Rt = pm.Deterministic

