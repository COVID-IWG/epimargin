import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib
from scipy.signal import convolve, deconvolve
from scipy.stats import gamma, logistic, poisson
from scipy.optimize import minimize 


color = [0.8423298817793848, 0.8737404427964184, 0.7524954030731037]

# x = np.linspace(0, 10)
# y = 100 * logistic.cdf(x - 6)
# z, _ = deconvolve(y, np.flip(filter_/filter_.sum()))
# # plt.plot(x, y, c ="black", linewidth = 3)
# plt.plot(x, y, c = color,  linewidth = 2)
# plt.plot(x, z, c = "black")
# plt.show()

palette = [[0.8423298817793848, 0.8737404427964184, 0.7524954030731037], [0.5815252468131623, 0.7703468311289211, 0.5923205247665932], [0.35935359003014994, 0.6245622005326175, 0.554154071059354], [0.25744332683867743, 0.42368146872794976, 0.5191691971789514], [0.21392162678343224, 0.20848424698401846, 0.3660805512579508]]


a = 5
t = np.linspace(0, 200)
f = 2000 * logistic.cdf((t - 75)/10)
orig = np.r_[np.zeros(50), f, f[-1] * np.ones(50), np.flip(f)]
pmf  = gamma.pdf(np.linspace(gamma.ppf(0.005, a), gamma.ppf(1-0.005, a)), a)
pmf/= sum(pmf)
obs  = convolve(orig, pmf, mode = "full")
obs *= sum(orig)/sum(obs)

plt.plot(obs,  color = palette[1], label="symptom onset reports", linewidth = 3)
plt.plot(orig, color = "black",    label="infections", linewidth = 3)
plt.xlabel("time")
plt.ylabel("cases")
plt.legend()
print(tikzplotlib.get_tikz_code())

# b = 3
# orig = np.r_[0, 4, 6, 9, 7, 5, np.zeros(14)]
# pmf = poisson.pmf(range(9), b)
# plt.plot(pmf)
# plt.show()

blur = convolve(orig, pmf, mode = "full")
plt.plot(orig)
plt.plot(blur)

plt.show()

# http://freerangestats.info/blog/2020/07/18/victoria-r-convolution
def deconv(observed, kernel): 
    k = len(kernel)
    padded = np.r_[np.zeros(k), observed]
    def error(x):
        return sum(convolve(x, kernel, mode="same")[:len(padded)] - padded) ** 2
    
    res = minimize(error, np.r_[observed, np.zeros(k)], method = "L-BFGS-B")
    return res.x


I_deconv, _ = deconvolve(obs, pmf)
# plt.plot(orig, label = "original")
plt.plot(obs, label = "observed")
plt.plot(I_deconv, label = "deconvolved")
plt.legend()
plt.show()