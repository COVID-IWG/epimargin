from tqdm.std import tqdm
import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import Age_SIRVD
from adaptive.utils import normalize, years, percent, annually
from scipy.stats import multinomial
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from studies.age_structure.wtp import *

sns.set(style = "whitegrid")

num_sims         = 1000
simulation_range = 1 * years
phi_points       = [_ * percent * annually for _ in (25, 50, 100)]
districts_to_run = district_IFR.filter(items = sorted(set(district_codes.keys()) - set(["Perambalur"])), axis = 0)
# districts_to_run = district_IFR.filter(items = ["Chennai", "Cuddalore", "Salem"], axis = 0)

def get_metrics(policy, counterfactual, district, N_district):
    f_hat_p1v1 = estimate_consumption_decline(district, 
        pd.Series(np.zeros(simulation_range + 1)), 
        pd.Series(np.zeros(simulation_range + 1)), force_natl_zero = True)
    c_p1v1 = (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values

    dI_pc_p1 = np.squeeze(policy.dT_total)/N_district
    dD_pc_p1 = np.squeeze(policy.dD_total)/N_district
    f_hat_p1v0 = np.array([estimate_consumption_decline(district, dI_pc_p1[:, _], dD_pc_p1[:, _]) for _ in range(num_sims)])
    c_p1v0 = (1 + f_hat_p1v0) * consumption_2019.loc[district].values[:, None, None]

    dI_pc_p0 = np.squeeze(counterfactual.dT_total)/N_district
    dD_pc_p0 = np.squeeze(counterfactual.dD_total)/N_district
    f_hat_p0v0 = np.array([estimate_consumption_decline(district, dI_pc_p0[:, _], dD_pc_p0[:, _]) for _ in range(num_sims)])
    c_p0v0 = (1 + f_hat_p0v0) * consumption_2019.loc[district].values[:, None, None]

    pi = np.squeeze(policy.pi) 
    q_p1v1 = np.squeeze(policy.q1)
    q_p1v0 = np.squeeze(policy.q0)
    q_p0v0 = np.squeeze(counterfactual.q0)
    
    WTP_daily_1 = pi * q_p1v1 * c_p1v1[:, None] + (1 - pi) * q_p1v0 * c_p1v0.T
    WTP_daily_0 = q_p0v0 * c_p0v0.T 
    dWTP_daily = WTP_daily_1 - WTP_daily_0

    dWTP_hlth_daily = \
        (1 - pi) * (q_p1v0 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v0.T +\
             pi  * (q_p1v1 - q_p0v0.mean(axis = 1)[:, None, :]) * c_p1v1[:, None]

    dWTP_cons_daily = \
        (1 - pi) * q_p0v0.mean(axis = 1)[:, None, :] * (c_p1v0 - c_p0v0.mean(axis = 1)[:, None, :]).T +\
             pi  * q_p1v1.mean(axis = 1)[:, None, :] * (c_p1v1 - c_p0v0.mean(axis = 1).T)[:, None, :]

    dWTP_priv_daily = \
        q_p1v1 * c_p1v1[:, None] - q_p1v0 * c_p1v0.T

    VSLY_daily_1 = ((1 - pi) * q_p1v0 + pi * q_p1v1) * np.mean(c_p1v0, axis = 1).T[:, None, :]
    VSLY_daily_0 = q_p0v0 * np.mean(c_p0v0, axis = 1).T[:, None, :]
    dVLSY_daily = VSLY_daily_1 - VSLY_daily_0

    beta = 1/(1 + 4.25/365)
    s = np.arange(simulation_range + 1)

    WTP      = [] 
    VSLY     = []
    WTP_priv = None
    WTP_hlth = None
    WTP_cons = None
    for t in range(simulation_range + 1):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * dWTP_daily[t:, :], axis = 0)
        WTP.append(wtp)

        vsly = np.sum(np.power(beta, s[t:])[:, None, None] * dVLSY_daily[t:, :], axis = 0)
        VSLY.append(vsly)

        if t == 0:
            WTP_hlth = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_hlth_daily[t:, :], axis = 0)
            WTP_cons = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_cons_daily[t:, :], axis = 0)
            WTP_priv = np.sum(np.power(beta, s[t:])[:, None, None] * dWTP_priv_daily[t:, :], axis = 0)

    WTP  = np.squeeze(WTP)
    VSLY = np.squeeze(VSLY)
    
    # health percentiles 
    # policy_death_percentiles         = np.percentile(policy.D[-1].sum(axis = 1), [50, 5, 95])
    # counterfactual_death_percentiles = np.percentile(counterfactual.D[-1].sum(axis = 1), [50, 5, 95])

    # policy_YLL_percentiles         = np.percentile(policy.D[-1] @ YLLs, [50, 5, 95])
    # counterfactual_YLL_percentiles = np.percentile(counterfactual.D[-1] @ YLLs, [50, 5, 95])

    # # WTP 
    # WTP_percentiles = np.percentile(WTPs[0, :, :], [50, 5, 95], axis = 0)
    # return WTPs, WTP_percentiles, policy_death_percentiles, policy_YLL_percentiles, counterfactual_death_percentiles, counterfactual_YLL_percentiles
    return (
        WTP,
        VSLY,
        (policy.D[-1] - policy.D[0]).sum(axis = 1),
        (policy.D[-1] - policy.D[0]) @ YLLs,
        (counterfactual.D[-1] - counterfactual.D[0]).sum(axis = 1),
        (counterfactual.D[-1] - counterfactual.D[0]) @ YLLs,
        WTP_hlth, 
        WTP_cons,
        WTP_priv
    )

def prioritize(num_doses, S, prioritization):
    Sp = S[:, prioritization]
    dV = np.where(Sp.cumsum(axis = 1) <= num_doses, Sp, 0)
    dV[np.arange(len(dV)), (Sp.cumsum(axis = 1) > dV.cumsum(axis = 1)).argmax(axis = 1)] = num_doses - dV.sum(axis = 1)
    return dV[:, [i for (i, _) in sorted(enumerate(prioritization), key = lambda t: t[1])]]

num_age_bins = 7

seed = 0

evaluated_WTP    = defaultdict(lambda: 0)
evaluated_WTP_h  = defaultdict(lambda: 0)
evaluated_WTP_c  = defaultdict(lambda: 0)
evaluated_WTP_p  = defaultdict(lambda: 0)
evaluated_VSLY   = defaultdict(lambda: 0)
evaluated_deaths = defaultdict(lambda: 0)
evaluated_YLL    = defaultdict(lambda: 0)

per_district_WTPs = {}

# age_district_percentiles = pd.DataFrame({
#     k: np.percentile(v[0, :, :], [50], axis = 0)[0]
#     for (k, v) in per_district_WTPs.items()
# }).T
# age_district_percentiles.columns = age_bin_labels

progress = tqdm(total = (simulation_range + 3) * len(districts_to_run) * len(phi_points))
for (district, seroprevalence, N_district, _, IFR_sero, _) in districts_to_run.itertuples():
    dT_conf_district = ts.loc[district].dT
    dT_conf_district = dT_conf_district.reindex(pd.date_range(dT_conf_district.index.min(), dT_conf_district.index.max()), fill_value = 0)
    dT_conf_district_smooth = pd.Series(smooth(dT_conf_district), index = dT_conf_district.index).clip(0).astype(int)
    T_conf_smooth = dT_conf_district_smooth.cumsum().astype(int)
    T = T_conf_smooth[date]
    T_sero = (N_district * seroprevalence)
    T_ratio = T_sero/T
    D0, R0 = ts.loc[district][["dD", "dR"]].sum()
    R0 *= T_ratio

    T_scaled = dT_conf_district_smooth.cumsum()[simulation_start] * T_ratio
    S0 = N_district - T_scaled
    dD0 = ts.loc[district].dD.loc[simulation_start]
    dT0 = ts.loc[district].dT.loc[simulation_start] * T_ratio
    I0 = max(0, (T_scaled - R0 - D0))

    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    def get_model(seed = 104):
        model = Age_SIRVD(
            name        = district, 
            population  = N_district - D0, 
            dT0         = np.ones(num_sims) * (dT_conf_district_smooth[simulation_start] * T_ratio).astype(int), 
            Rt0         = Rt[simulation_start],
            S0          = np.tile((fS * S0).T, num_sims).reshape((num_sims, -1)),
            I0          = np.tile((fI * I0).T, num_sims).reshape((num_sims, -1)),
            R0          = np.tile((fR * R0).T, num_sims).reshape((num_sims, -1)),
            D0          = np.tile((fD * D0).T, num_sims).reshape((num_sims, -1)),
            mortality   = np.array(list(TN_IFRs.values())), #(ts.loc[district].dD.cumsum()[simulation_start]/T_scaled if I0 == 0 else dD0/(gamma * I0)),
            random_seed = seed
        )
        model.dD_total[0] = np.ones(num_sims) * dD0
        model.dT_total[0] = np.ones(num_sims) * dT0
        return model

    for phi in phi_points:
        progress.set_description(f"{district:15s}(φ = {str(int(365 * 100 * phi)):>3s}%)")
        num_doses = phi * (S0 + I0 + R0)
        random_model, mortality_model, contact_model, no_vax_model = [get_model(seed) for _ in range(4)]
        for t in range(simulation_range):
            if t <= 1/phi:
                dV_random    = multinomial.rvs(num_doses, normalize(random_model.N[-1], axis = 1)[0]).reshape((-1, 7))
                dV_mortality = prioritize(num_doses, mortality_model.S[-1], [6, 5, 4, 3, 2, 1, 0]) 
                dV_contact   = prioritize(num_doses, contact_model.S[-1],   [1, 2, 3, 4, 0, 5, 6]) 
            else: 
                dV_random, dV_mortality, dV_contact = np.zeros((num_sims, 7)), np.zeros((num_sims, 7)), np.zeros((num_sims, 7))
            
            random_model.parallel_forward_epi_step(dV_random, num_sims = num_sims)
            mortality_model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)
            contact_model.parallel_forward_epi_step(dV_contact, num_sims = num_sims)
            no_vax_model.parallel_forward_epi_step(dV = np.zeros((7, num_sims))[:, 0], num_sims = num_sims)
            progress.update(1)

        random_wtps, random_vsly, random_deaths, random_ylls, no_vax_deaths, no_vax_ylls, wtp_hlth, wtp_cons, wtp_priv\
            = get_metrics(random_model, no_vax_model, district, N_district)
        progress.update(1)
        evaluated_WTP[phi * 365, "random"] += random_wtps
        evaluated_VSLY[phi * 365, "random"] += random_vsly
        evaluated_deaths[phi * 365, "random"] += random_deaths

        evaluated_YLL[phi * 365, "random"] += random_ylls
        evaluated_deaths[phi * 365, "no_vax"] += no_vax_deaths
        evaluated_YLL[phi * 365, "no_vax"] += no_vax_ylls
        
        per_district_WTPs[district] = random_wtps
        evaluated_WTP_h[district]  += wtp_hlth
        evaluated_WTP_c[district]  += wtp_cons
        evaluated_WTP_p[district]  += wtp_priv

        
        mortality_wtps, mortality_vsly, mortality_deaths, mortality_ylls, *_\
            = get_metrics(mortality_model, no_vax_model, district, N_district)
        progress.update(1)
        evaluated_WTP[phi * 365, "mortality"]    += mortality_wtps
        evaluated_VSLY[phi * 365, "mortality"]   += mortality_vsly
        evaluated_deaths[phi * 365, "mortality"] += mortality_deaths
        evaluated_YLL[phi * 365, "mortality"]    += mortality_ylls
        
        contact_wtps, contact_vsly, contact_deaths, contact_ylls, *_\
            = get_metrics(contact_model, no_vax_model, district, N_district)
        progress.update(1)
        evaluated_WTP[phi * 365, "contact"]    += contact_wtps
        evaluated_VSLY[phi * 365, "contact"]   += contact_vsly
        evaluated_deaths[phi * 365, "contact"] += contact_deaths
        evaluated_YLL[phi * 365, "contact"]    += contact_ylls

death_percentiles = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_deaths.items()}
YLL_percentiles   = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_YLL.items()}
VSLY_percentiles  = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_VSLY.items()}
WTP_percentiles   = {tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_WTP.items()}

# death outcomes 
#region
fig = plt.gcf()

md, lo, hi = death_percentiles[(0.25, "no_vax")]
*_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

##################

md, lo, hi = death_percentiles[(0.25, "random")]
*_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(0.25, "contact")]
*_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(0.25, "mortality")]
*_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

##################

md, lo, hi = death_percentiles[(0.50, "random")]
*_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(0.50, "contact")]
*_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(0.50, "mortality")]
*_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

##################

md, lo, hi = death_percentiles[(1.0, "random")]
*_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(1.0, "contact")]
*_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = death_percentiles[(1.0, "mortality")]
*_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.xticks([0, 1, 2, 3], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().ylabel("deaths\n")
plt.show()
# #endregion

# # YLL 
# #region
fig = plt.figure()

md, lo, hi = YLL_percentiles[(0.25, "no_vax")]
*_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.25, "random")]
*_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.25, "contact")]
*_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.25, "mortality")]
*_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.50, "random")]
*_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.50, "contact")]
*_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(0.50, "mortality")]
*_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]


md, lo, hi = YLL_percentiles[(1.0, "random")]
*_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(1.0, "contact")]
*_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = YLL_percentiles[(1.0, "mortality")]
*_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.xticks([0, 1, 2, 3], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().ylabel("YLLs\n")
plt.show()
#endregion

# WTP
#region
fig = plt.figure()

# md, lo, hi = WTP_percentiles[(0.25, "no_vax")]
# *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.25, "random")] * USD
*_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.25, "contact")] * USD
*_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.25, "mortality")] * USD
*_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.50, "random")] * USD
*_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.50, "contact")] * USD
*_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(0.50, "mortality")] * USD
*_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(1.0, "random")] * USD
*_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(1.0, "contact")] * USD
*_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = WTP_percentiles[(1.0, "mortality")] * USD
*_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.xticks([1, 2, 3], ["$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().ylabel("WTP (USD)\n")
plt.show()
#endregion

# VSLY 
#region
fig = plt.figure()

# md, lo, hi = WTP_percentiles[(0.25, "no_vax")]
# *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.25, "random")] * USD
*_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.25, "contact")] * USD
*_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.25, "mortality")] * USD
*_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.50, "random")] * USD
*_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.50, "contact")] * USD
*_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(0.50, "mortality")] * USD
*_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(1.0, "random")] * USD
*_, bars = plt.errorbar(x = [3 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(1.0, "contact")] * USD
*_, bars = plt.errorbar(x = [3], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

md, lo, hi = VSLY_percentiles[(1.0, "mortality")] * USD
*_, bars = plt.errorbar(x = [3 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
[_.set_alpha(0.5) for _ in bars]

plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.xticks([1, 2, 3], ["$\phi = 25$%", "$\phi = 50$%", "$\phi = 100$%"], fontsize = "20")
plt.yticks(fontsize = "20")
plt.PlotDevice().ylabel("VSLY (USD)\n")
plt.show()
#endregion

# aggregate WTP by age
WTP_random_50_percentile = np.percentile(evaluated_WTP[0.5, "random"][0, :, :], [50, 5, 95], axis = 0) * USD
fig = plt.figure()
for (i, (md, lo, hi)) in enumerate(WTP_random_50_percentile.T):
    *_, bars = plt.errorbar(x = [i], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
    fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
    [_.set_alpha(0.5) for _ in bars]
plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "20")
plt.yticks(fontsize = "20")
plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
plt.PlotDevice().ylabel("aggregate WTP (USD)\n")
plt.show()

print("")

# health/consumption
summed_wtp_health = np.median(np.sum([np.array(_) for _ in evaluated_WTP_h.values()], axis = 0), axis = 0)
summed_wtp_income = np.median(np.sum([np.array(_) for _ in evaluated_WTP_c.values()], axis = 0), axis = 0)
fig, ax = plt.subplots()
ax.bar(range(7), summed_wtp_income * USD, bottom = summed_wtp_health * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2)
ax.bar(range(7), summed_wtp_health * USD,                                   color = age_group_colors, edgecolor = age_group_colors, linewidth = 2)
ax.bar(range(7), [0], label = "income", color = "white", edgecolor = "black", linewidth = 2)
ax.bar(range(7), [0], label = "health", color = "black", edgecolor = "black", linewidth = 2)
plt.xticks(range(7), age_bin_labels, fontsize = "20")
plt.yticks(fontsize = "20")
plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.PlotDevice().ylabel("WTP (USD)\n")
plt.semilogy()
plt.show()

# social/private 
summed_wtp_priv = np.median(np.sum([np.array(_) for _ in evaluated_WTP_p.values()], axis = 0), axis = 0)
summed_wtp_tot  = np.median(np.sum([np.array(_) for _ in per_district_WTPs.values()], axis = 0)[0, :, :], axis = 0)
# summed_wtp      = np.median(np.sum([np.array(_) for _ in evaluated_WTP  .values()], axis = 0)[0, :, :], axis = 0)
summed_wtp_soc  = summed_wtp_tot - summed_wtp_priv
fig, ax = plt.subplots()
ax.bar(range(7), summed_wtp_priv * USD, bottom = summed_wtp_soc * USD, color = "white",          edgecolor = age_group_colors, linewidth = 2)
ax.bar(range(7), summed_wtp_soc  * USD,                                color = age_group_colors, edgecolor = age_group_colors, linewidth = 2)
ax.bar(range(7), [0], label = "social",  color = "white", edgecolor = "black", linewidth = 2)
ax.bar(range(7), [0], label = "private", color = "black", edgecolor = "black", linewidth = 2)

plt.xticks(range(7), age_bin_labels, fontsize = "20")
plt.yticks(fontsize = "20")
plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
plt.PlotDevice().ylabel("WTP (USD)\n")
plt.semilogy()
plt.show()

# dist x age 
per_district_percentiles = {district: np.percentile(wtp[0, :, :], [50, 5, 95], axis = 0) for (district, wtp) in per_district_WTPs.items()}

fig = plt.figure()
district_ordering = list(per_district_percentiles.keys())[:5]
for (i, district) in enumerate(district_ordering):
    wtps = per_district_percentiles[district]
    for j in range(7):
        plt.errorbar(
            x = [i + 0.1 * (j - 3)],
            y = wtps[0, 6-j] * USD,
            yerr = [
                [(wtps[0, 6-j] - wtps[1, 6-j]) * USD],
                [(wtps[2, 6-j] - wtps[0, 6-j]) * USD]
            ], 
            fmt = "o",
            color = age_group_colors[6-j],
            figure = fig,
            label = None if i > 0 else age_bin_labels[6-j]
        )
plt.xticks(
    range(len(district_ordering)),
    district_ordering,
    rotation = 45,
    fontsize = "20"
)
plt.yticks(fontsize = "20")
plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
# plt.ylim(0, 10000)
plt.xlim(-0.5, len(district_ordering) - 0.5)
plt.PlotDevice().ylabel("WTP (USD)\n")
plt.show()