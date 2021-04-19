import pandas as pd 

from epimargin.smoothing import notched_smoothing
from epimargin.estimators import analytical_MPVS
import epimargin.plots as plt 

CI = 0.95
gamma = 0.2
window = 3
smoothing = notched_smoothing(window = window)

schema = { 
    "Date Symptom Onset"          : "symptom_onset",
    "Date of Hospital  Admissions": "admission",
    "Date tested"                 : "tested",
    "Date of positive test result": "confirmed",
    "Date Recovered"              : "recovered",
    "Date Died"                   : "died",
    "Kebupaten/Kota"              : "regency",
    "Kecamatan"                   : "district", 
    "age "                        : "age"
}

regency_names = { 
    'Pangkep'  : 'Pangkajene Dan Kepulauan', 
    'Pare-Pare': 'Parepare', 
    'Selayar'  : 'Kepulauan Selayar', 
    'Sidrap'   : 'Sidenreng Rappang'
}

def parse_datetimes(df):
    valid_idx = ~df.isna() & df.str.endswith("20")
    valid = df[valid_idx]
    monthfirst_idx = valid.str.endswith("/20") # short years -> month first notation 
    valid.loc[( monthfirst_idx)] = pd.to_datetime(valid[( monthfirst_idx)], errors = 'coerce', format = "%m/%d/%y", dayfirst = False)
    valid.loc[(~monthfirst_idx)] = pd.to_datetime(valid[(~monthfirst_idx)], errors = 'coerce', format = "%d/%m/%Y", dayfirst = True)
    # assert df.max() <= pd.to_datetime("October 03, 2020"), "date parsing resulted in future dates"
    df.loc[valid_idx] = valid.apply(pd.Timestamp)

def parse_age(s, bound = 100):
    try: 
        return min(bound, int(float(s.strip().split(" ")[0])))
    except:
        return None

cases = pd.read_csv("data/3 OCT 2020 Data collection template update South Sulawesi_CASE.csv", usecols = schema.keys())\
        .rename(columns = schema)\
        .dropna(how = 'all')\
        .query("age.str.strip() != ''", engine = "python")
parse_datetimes(cases.loc[:, "confirmed"])
cases.regency = cases.regency.str.title().map(lambda s: regency_names.get(s, s))
cases.age     = cases.age.apply(parse_age)
cases = cases.dropna(subset = ["age"])
cases["age_bin"] = pd.cut(cases.age, bins = [0] + list(range(20, 80, 10)) + [100])
age_ts = cases[["age_bin", "confirmed"]].groupby(["age_bin", "confirmed"]).size().sort_index()
ss_max_rts = {}

fig, axs = plt.subplots(4, 2, True, True)
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(age_ts.sum(level = 1), CI = CI, smoothing = notched_smoothing(window = 5), totals = False)
plt.sca(axs.flat[0])
plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI).annotate(f"all ages").adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.05, hspace = 0.3, wspace = 0.15)
r = pd.Series(Rt_pred, index = dates)
ss_max_rts["all"] = r[r.index.month_name() == "April"].max()

for (age_bin, ax) in zip(age_ts.index.get_level_values(0).categories, axs.flat[1:]):
    print(age_bin)
    (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
        = analytical_MPVS(age_ts.loc[age_bin], CI = CI, smoothing = smoothing, totals = False)
    plt.sca(ax)
    plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI).annotate(f"age bin: {age_bin}")
    ax.get_legend().remove()
    r = pd.Series(Rt_pred, index = dates)
    ss_max_rts[age_bin] = r[r.index.month_name() == "April"].max()
plt.show()

makassar_age_ts = cases.query("regency == 'Makassar'")[["age_bin", "confirmed"]].groupby(["age_bin", "confirmed"]).size().sort_index()
mak_max_rts = {}

fig, axs = plt.subplots(4, 2, True, True)
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(makassar_age_ts.sum(level = 1), CI = CI, smoothing = smoothing, totals = False)
plt.sca(axs.flat[0])
plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI).annotate(f"all ages").adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.05, hspace = 0.3, wspace = 0.15)
r = pd.Series(Rt_pred, index = dates)
mak_max_rts["all"] = r[r.index.month_name() == "April"].max()

for (age_bin, ax) in zip(makassar_age_ts.index.get_level_values(0).categories, axs.flat[1:]):
    print(age_bin)
    (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
        = analytical_MPVS(makassar_age_ts.loc[age_bin], CI = CI, smoothing = smoothing, totals = False)
    plt.sca(ax)
    plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI).annotate(f"age bin: {age_bin}")
    ax.get_legend().remove()
    r = pd.Series(Rt_pred, index = dates)
    mak_max_rts[age_bin] = r[r.index.month_name() == "April"].max()
plt.show()
