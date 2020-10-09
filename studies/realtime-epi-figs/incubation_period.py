# from scipy.stats import gamma, lognorm
import scipy.stats 
import matplotlib.pyplot as plt
import numpy as np 
import tikzplotlib

color = "#AFC581" #[0.8423298817793848, 0.8737404427964184, 0.7524954030731037]
alpha = 0.0001

def get_pdf(dist, lb = alpha/2, ub = 1 - (alpha/2), nq = 1000):
    qnt = np.linspace(dist.ppf(lb), dist.ppf(ub), nq)
    return (qnt, dist.pdf(qnt))

# meta analysis: https://bmjopen.bmj.com/content/10/8/e039652
# lognormal params
mu    = 1.63
sigma = 0.50
mean  = 5.8

lognorm = scipy.stats.lognorm(scale = np.exp(mu), s = sigma)
gamma   = scipy.stats.gamma(lognorm.mean())

ln_pdf = get_pdf(lognorm)
gm_pdf = get_pdf(gamma)

ln_l, = plt.plot(*ln_pdf, color = "black", alpha = 0.6)
ln_a  = plt.fill_between(*ln_pdf, color = "black", alpha = 0.1, label = "log-normal")
gm_l, = plt.plot(*gm_pdf, color = color)
gm_a  = plt.fill_between(*gm_pdf, color = color, alpha = 0.5, label = "gamma")
plt.xlim(left = 0, right = 20)
plt.ylim(bottom = 0)
plt.legend()
plt.xlabel("days")
plt.ylabel("probability density for generation interval")
tikzplotlib.clean_figure()
tikzplotlib.save("gen_int.tex")
plt.show()

