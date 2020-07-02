import matplotlib.pyplot as plt
import pandas as pd

from adaptive.estimators import box_filter, gamma_prior
from adaptive.plots import plot_RR_est, plot_T_anomalies

# model details
CI        = 0.95
smoothing = 21

true_Rt   = pd.read_table("./true_Rt.txt",   dtype="float", squeeze=True)
obs_cases = pd.read_table("./obs_cases.txt", dtype="float", squeeze=True)

(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = gamma_prior(obs_cases, CI = CI, smoothing = lambda ts: box_filter(ts, smoothing, smoothing//2))

print("  + Rt today:", RR_pred[-1])

plot_RR_est(dates, RR_pred, RR_CI_lower, RR_CI_upper, CI)\
    .ylabel("Estimated $R_t$")\
    .title("Synthetic Data Estimation")\
    .size(11, 8)
plt.plot(true_Rt.index, true_Rt.values, 'k--', label="True $R_t$")
plt.xlim(0, 150)
plt.ylim(0, 2.5)
plt.legend()
# plt.savefig("synthetic_Rt_est.png", dpi=600, bbox_inches="tight")
plt.show()