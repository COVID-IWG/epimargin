import adaptive.plots as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adaptive.estimators import analytical_MPVS
from adaptive.models import Age_SIRVD
from adaptive.utils import normalize
from scipy.stats import multinomial
from studies.age_structure.commons import *
from studies.age_structure.palette import *
from studies.age_structure.wtp import *

sns.set(style = "whitegrid")

num_sims = 100
def get_metrics(policy, counterfactual, district = "Chennai"):
    f_hat_p1v1 = estimate_consumption_decline(district, 
        pd.Series(np.zeros(366)), 
        pd.Series(np.zeros(366)), force_natl_zero = True)
    c_p1v1 = (1 + f_hat_p1v1)[:, None] * consumption_2019.loc[district].values

    dI_pc_p1 = pd.DataFrame(np.sum(np.squeeze(policy.I) + np.squeeze(policy.I_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    dD_pc_p1 = pd.DataFrame(np.sum(np.squeeze(policy.D) + np.squeeze(policy.D_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    f_hat_p1v0 = np.array([estimate_consumption_decline(district, dI_pc_p1.iloc[:, _], dD_pc_p1.iloc[:, _]) for _ in range(num_sims)])
    c_p1v0 = (1 + f_hat_p1v0) * consumption_2019.loc[district].values[:, None, None]

    dI_pc_p0 = pd.DataFrame(np.sum(np.squeeze(counterfactual.I) + np.squeeze(counterfactual.I_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    dD_pc_p0 = pd.DataFrame(np.sum(np.squeeze(counterfactual.D) + np.squeeze(counterfactual.D_vn), axis = 2)/N_district).diff().shift(-1).fillna(0)
    f_hat_p0v0 = np.array([estimate_consumption_decline(district, dI_pc_p0.iloc[:, _], dD_pc_p0.iloc[:, _]) for _ in range(num_sims)])
    c_p0v0 = (1 + f_hat_p0v0) * consumption_2019.loc[district].values[:, None, None]

    pi = np.squeeze(policy.pi) 
    q_p1v1 = np.squeeze(policy.q1)
    q_p1v0 = np.squeeze(policy.q0)
    q_p0v0 = np.squeeze(counterfactual.q0)
    
    WTP_daily_1 = pi * q_p1v1 * c_p1v1[:, None] + (1 - pi) * q_p1v0 * c_p1v0.T
    WTP_daily_0 = q_p0v0 * c_p0v0.T 

    beta = 1/(1 + 4.25/365)
    s = np.arange(366)

    dWTP_daily = WTP_daily_1 - WTP_daily_0

    WTPs   = [] 
    for t in range(366):
        wtp = np.sum(np.power(beta, s[t:] - t)[:, None, None] * dWTP_daily[t:, :], axis = 0)
        WTPs.append(wtp)
        # if t == 0:
        #     wtp_h = (1 - pi) * (q_p1v0 - q_p0v0) * c_p1v0.T + pi * (q_p1v1 - q_p0v0) * c_p1v1[:, None]
    WTPs = np.squeeze(WTPs)
    
    # health percentiles 
    # policy_death_percentiles         = np.percentile(policy.D[-1].sum(axis = 1), [50, 5, 95])
    # counterfactual_death_percentiles = np.percentile(counterfactual.D[-1].sum(axis = 1), [50, 5, 95])

    # policy_YLL_percentiles         = np.percentile(policy.D[-1] @ YLLs, [50, 5, 95])
    # counterfactual_YLL_percentiles = np.percentile(counterfactual.D[-1] @ YLLs, [50, 5, 95])

    # # WTP 
    # WTP_percentiles = np.percentile(WTPs[0, :, :], [50, 5, 95], axis = 0)
    # return WTPs, WTP_percentiles, policy_death_percentiles, policy_YLL_percentiles, counterfactual_death_percentiles, counterfactual_YLL_percentiles
    return WTPs, policy.D[-1].sum(axis = 1), policy.D[-1] @ YLLs, counterfactual.D[-1].sum(axis = 1), counterfactual.D[-1] @ YLLs

def prioritize(num_doses, S, prioritization):
    Sp = S[:, prioritization]
    dV = np.where(Sp.cumsum(axis = 1) <= num_doses, Sp, 0)
    dV[np.arange(len(dV)), (Sp.cumsum(axis = 1) > dV.cumsum(axis = 1)).argmax(axis = 1)] = num_doses - dV.sum(axis = 1)
    return dV[:, [i for (i, _) in sorted(enumerate(prioritization), key = lambda t: t[1])]]

num_age_bins = 7


across_bins = dict(axis = 0)
across_time = dict(axis = 1)


seed = 0

evaluated_WTPs   = defaultdict(lambda: 0)
evaluated_deaths = defaultdict(lambda: 0)
evaluated_YLLs   = defaultdict(lambda: 0)

per_district_WTPs = {}

age_district_percentiles = pd.DataFrame({
    k: np.percentile(v[0, :, :], [50], axis = 0)[0]
    for (k, v) in per_district_WTPs.items()
}).T
age_district_percentiles.columns = age_bin_labels

for (district, seroprevalence, N_district, _, IFR_sero, _) in district_IFR.filter(items = sorted(set(district_codes.keys()) - set(["Perambalur"])), axis = 0).itertuples():
    print(district)
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
    I0 = max(0, (T_scaled - R0 - D0))

    (Rt_dates, Rt_est, *_) = analytical_MPVS(T_ratio * dT_conf_district_smooth, CI = CI, smoothing = lambda _:_, totals = False)
    Rt = dict(zip(Rt_dates, Rt_est))

    def get_model(seed = 104):
        return Age_SIRVD(
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
    for phi in (0.25/365, 0.5/365):
        num_doses = phi * (S0 + I0 + R0)
        random_model, mortality_model, contact_model, no_vax_model = [get_model(seed) for _ in range(4)]
        # dVs = {_:[] for _ in ("random", "mortality", "contact")}

        for t in range(1 * 365):
            if t <= 1/phi:
                dV_random    = multinomial.rvs(num_doses, normalize(random_model.N[-1], axis = 1)[0]).reshape((-1, 7))
                dV_mortality = prioritize(num_doses, mortality_model.S[-1], [6, 5, 4, 3, 2, 1, 0]) 
                dV_contact   = prioritize(num_doses, contact_model.S[-1], [1, 2, 3, 4, 0, 5, 6]) 
            else: 
                dV_random, dV_mortality, dV_contact = np.zeros((num_sims, 7)), np.zeros((num_sims, 7)), np.zeros((num_sims, 7))
            # dVs["random"].append(dV_random)
            # dVs["mortality"].append(dV_mortality)
            # dVs["contact"].append(dV_contact)
            
            random_model.parallel_forward_epi_step(dV_random, num_sims = num_sims)
            mortality_model.parallel_forward_epi_step(dV_mortality, num_sims = num_sims)
            contact_model.parallel_forward_epi_step(dV_contact, num_sims = num_sims)
            no_vax_model.parallel_forward_epi_step(dV = np.zeros((7, num_sims))[:, 0], num_sims = num_sims)

        random_wtps, random_deaths, random_ylls, no_vax_deaths, no_vax_ylls\
            = get_metrics(random_model, no_vax_model, district)
        evaluated_deaths[phi * 365, "no_vax"] += no_vax_deaths
        evaluated_YLLs[phi * 365, "no_vax"] += no_vax_ylls
        evaluated_WTPs[phi * 365, "random"] += random_wtps
        evaluated_deaths[phi * 365, "random"] += random_deaths
        evaluated_YLLs[phi * 365, "random"] += random_ylls
        per_district_WTPs[district] = random_wtps
        
        mortality_wtps, mortality_deaths, mortality_ylls, *_\
            = get_metrics(mortality_model, no_vax_model, district)
        evaluated_WTPs[phi * 365, "mortality"] += mortality_wtps
        evaluated_deaths[phi * 365, "mortality"] += mortality_deaths
        evaluated_YLLs[phi * 365, "mortality"] += mortality_ylls
        
        contact_wtps, contact_deaths, contact_ylls, *_\
            = get_metrics(contact_model, no_vax_model, district)
        evaluated_WTPs[phi * 365, "contact"] += contact_wtps
        evaluated_deaths[phi * 365, "contact"] += contact_deaths
        evaluated_YLLs[phi * 365, "contact"] += contact_ylls
        
        # WTPs[phi * 365, "random"] = random_WTP_percentiles
        # WTPs[phi * 365, "mortality"] = mortality_WTP_percentiles
        # WTPs[phi * 365, "contact"] = contact_WTP_percentiles

        # deaths[phi * 365, "random"] = random_death_percentiles
        # deaths[phi * 365, "mortality"] = mortality_death_percentiles
        # deaths[phi * 365, "contact"] = contact_death_percentiles
        # deaths[phi * 365, "no_vax"] = no_vax_death_percentiles

        # YLLs[phi * 365, "random"] = random_YLL_percentiles
        # YLLs[phi * 365, "mortality"] = mortality_YLL_percentiles
        # YLLs[phi * 365, "contact"] = contact_YLL_percentiles
        # YLLs[phi * 365, "no_vax"] = no_vax_YLL_percentiles

death_percentiles ={tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_deaths.items()}
YLL_percentiles ={tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_YLLs.items()}
WTP_percentiles ={tag: np.percentile(metric, [50, 5, 95]) for (tag, metric) in evaluated_WTPs.items()}
# death outcomes 
#region
# fig = plt.figure()

# md, lo, hi = death_percentiles[(0.25, "no_vax")]
# *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.25, "random")]
# *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.25, "contact")]
# *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.25, "mortality")]
# *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.50, "random")]
# *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.50, "contact")]
# *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = death_percentiles[(0.50, "mortality")]
# *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
# plt.xticks([0, 1, 2], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%"], fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.PlotDevice().ylabel("deaths\n")
# plt.show()
# #endregion

# # YLL 
# #region
# fig = plt.figure()

# md, lo, hi = YLL_percentiles[(0.25, "no_vax")]
# *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.25, "random")]
# *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.25, "contact")]
# *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.25, "mortality")]
# *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.50, "random")]
# *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = random_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.50, "contact")]
# *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = contactrate_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = YLL_percentiles[(0.50, "mortality")]
# *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "o", color = mortality_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
# plt.xticks([0, 1, 2], ["$\phi = 0$%", "$\phi = 25$%", "$\phi = 50$%"], fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.PlotDevice().ylabel("YLLs\n")
# plt.show()
# #endregion

# # WTP
# #region
# fig = plt.figure()

# # md, lo, hi = WTP_percentiles[(0.25, "no_vax")]
# # *_, bars = plt.errorbar(x = [0], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
# #     fmt = "D", color = no_vax_color, label = "no vaccination", ms = 12, elinewidth = 5)
# # [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.25, "random")] * USD
# *_, bars = plt.errorbar(x = [1 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = random_vax_color, label = "random assignment", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.25, "contact")] * USD
# *_, bars = plt.errorbar(x = [1], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = contactrate_vax_color, label = "contact rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.25, "mortality")] * USD
# *_, bars = plt.errorbar(x = [1 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = mortality_vax_color, label = "mortality rate prioritized", ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.50, "random")] * USD
# *_, bars = plt.errorbar(x = [2 - 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = random_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.50, "contact")] * USD
# *_, bars = plt.errorbar(x = [2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = contactrate_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# md, lo, hi = WTP_percentiles[(0.50, "mortality")] * USD
# *_, bars = plt.errorbar(x = [2 + 0.2], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = mortality_vax_color, ms = 12, elinewidth = 5)
# [_.set_alpha(0.5) for _ in bars]

# plt.legend(ncol = 4, fontsize = "20", loc = "lower center", bbox_to_anchor = (0.5, 1))
# plt.xticks([1, 2], ["$\phi = 25$%", "$\phi = 50$%"], fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.PlotDevice().ylabel("WTP (USD)\n")
# plt.show()
# #endregion

# # WTP by age
# WTP_random_50_percentile = np.percentile(evaluated_WTPs[0.5, "random"].sum(axis = 0), [50, 5, 95], axis = 0) * USD
# fig = plt.figure()
# for (i, (md, lo, hi)) in enumerate(WTP_random_50_percentile.T):
#     *_, bars = plt.errorbar(x = [i], y = [md], yerr = [[md - lo], [hi - md]], figure = fig,
#     fmt = "D", color = age_group_colors[i], ms = 12, elinewidth = 5, label = age_bin_labels[i])
#     [_.set_alpha(0.5) for _ in bars]
# plt.xticks([0, 1, 2, 3, 4, 5, 6], age_bin_labels, fontsize = "20")
# plt.yticks(fontsize = "20")
# plt.legend(title = "age bin", title_fontsize = "20", fontsize = "20")
# plt.PlotDevice().ylabel("1-year aggregate WTP (USD)\n")
# plt.show()
