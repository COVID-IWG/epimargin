import adaptive.plots as plt
import pandas as pd
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.models import SIR, NetworkedSIR
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

CI = 0.95
gamma = 0.2
window = 3
smoothing = notched_smoothing(window = window)

replacements = { 
    "CAKUNG":           ['CAKUNG'],
    "CEMPAKA PUTIH":    ['CEMPA PUTIH', 'CEMPAKA PUTIH'],
    "CENGKARENG":       ['CENGAKRENG', 'CENGKARENG'],
    "CILANDAK":         ['CIILANDAK', 'CILANDAK'],
    "CILINCING":        ['CILINCIN','CILINCING', 'CILNCING', 'CLILINCING'],
    "CIPAYUNG":         ['CIPAYUNG'],
    "CIRACAS":          ['CIRACAS'],
    "DUREN SAWIT":      ['DUREN SAWIT'],
    "GAMBIR":           ['GAMBIR'],
    "GROGOLPETAMBURAN": ['GROGOL','GROGOL PEATAMBURAN', 'GROGOL PETAMBURAN', 'GROGOL PETEMBURAN', 'PETAMBURAN'],
    "JAGAKARSA":        ['JAGAKARSA'],
    "JATINEGARA":       ['JATIENGARA', 'JATINEGARA', 'JATINEGARAA'],
    "JOHAR BARU":       ['JOHAR BARU', 'JOHOR BARU'],
    "KALIDERES":        ['KALI DERES', 'KALIDERES'],
    "KEBAYORAN BARU":   ['KEBAYORAN BARU'],
    "KEBAYORAN LAMA":   ['KEBAYORAN LAMA', 'KEBAYORAN LAMA SELATAN', 'KEBAYORAN LIMA', 'KEBYORAN LAMA'],
    "KEBONJERUK":       ['KEBONJERUK', 'KEBON JERUK', 'KEBON JEURK'],
    "KELAPA GADING":    ['KELAPA GADING', 'KRELAPA GADING'],
    "KEMAYORAN":        ['KEMAYORAN'],
    "KEMBANGAN":        ['KEMBAGAN', 'KEMBANGAN'],
    "KOJA":             ['KOJA'],
    "KRAMATJATI":       ['KRAMAT JATI', 'KRAMATJATI'],
    "MAKASAR":          ['MAKASAR', 'MAKASSAR'],
    "MAMPANG PRAPATAN": ['MAMPANG', 'MAMPANG PERAPATAN', 'MAMPANG PRAPATAN'],
    "MATRAMAN":         ['MATRAMAN'],
    "MENTENG":          ["MENTENG"],
    "PADEMANGAN":       ['PADEMANGAN'],
    "PALMERAH":         ['PALMERAH'],
    "PANCORAN":         ['PANCORAN'],
    "PASAR MINGGU":     ['PASAR MINGGU', 'PAASAR MINGGU'],
    "PASARREBO":        ['PASARREBO', 'PASAR REBO'],
    "PENJARINGAN":      ['PENJAGALAN', 'PENJARINGAN', 'PENJARINGAN UTARA'],
    "PESANGGRAHAN":     ['PESANGGRAHAN'],
    "PULOGADUNG":       ['PULO GADUNG', 'PULOGADUNG'],
    "SAWAH BESAR":      ['SAWAH BESAR'],
    "SENEN":            ['SENEN'],
    "SETIA BUDI":       ['SETIA BUDI', 'SETIA BUSI', 'SEIA BUDI'],
    "SETIABUDI":        ['SETIABUDI'],
    "TAMANSARI":        ['TAMAN SARI', 'TAMANSARI'],
    "TAMBORA":          ['TAMBORA'],
    "TANAHABANG":       ['TANAHABANG', 'TANAH ABANG'],
    "TANJUNG PRIOK":    ['TAMAN SARI', 'TAMANSARI'],
    "TEBET":            ['TEBET'],
}


dkij_drop_cols = [
    'age', 'sex', 'fever', 'temp', 'cough', 'flu', 'sore_throat', 'shortness_breath', 'shivering', 'headache', 'malaise', 'muscle_pain',
    'nausea_vomiting', 'abdominal_pain', 'diarrhoea', 'date_recovered',
    'date_died', 'heart_disease', 'diabetes', 'pneumonia', 'hypertension', 'malignant',
    'immunology_disorder', 'chronic_kidney', 'chronic_liver', 'copd',
    'obesity', 'pregnant', 'tracing', 'otg', 'icu', 'intubation', 'ecmo',
    'criteria_cases', 'age_group', 'age_group2', 'date_discharge',
    'patient_status', 'death'
][1:] # keep age

(data, figs) = setup()

cases = pd.read_stata(data/"coviddkijakarta_290920.dta")\
        .query("province == 'DKI JAKARTA'")\
        .drop(columns = dkij_drop_cols + ["province"])
cases = cases\
    .set_axis(cases.columns.str.lower(), 1)\
    .assign(
        district    = cases.district.str.title(),
        subdistrict = cases.subdistrict.apply(lambda name: next((k for (k, v) in replacements.items() if name in v), name)), 
    )

cases["age_bin"] = pd.cut(cases.age, bins = [0] + list(range(20, 80, 10)) + [100])
age_ts = cases[["age_bin", "date_positiveresult"]].groupby(["age_bin", "date_positiveresult"]).size().sort_index()
dkij_max_rts = {}

(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(age_ts.sum(level = 1), CI = CI, smoothing = smoothing, totals = False)
r = pd.Series(Rt_pred, index = dates)
dkij_max_rts["all"] = r[r.index.month_name() == "April"].max()

for age_bin in age_ts.index.get_level_values(0).categories:
    print(age_bin)
    (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
        = analytical_MPVS(age_ts.loc[age_bin], CI = CI, smoothing = smoothing, totals = False)
    r = pd.Series(Rt_pred, index = dates)
    dkij_max_rts[age_bin] = r[r.index.month_name() == "April"].max()

print(dkij_max_rts)