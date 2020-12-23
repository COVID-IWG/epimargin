import json
from logging import getLogger
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from tqdm import tqdm

import adaptive.plots as plt
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup


# public cases 

date_scale  = 1000000.0
date        = "tanggal"
timeseries  = "list_perkembangan"
total_cases = "AKUMULASI_KASUS"
provinces = ["DKI JAKARTA", "SULAWESI SELATAN"]

filename = lambda province: "prov_detail_{}.json".format(province.replace(" ", "_"))

def load_province_timeseries(data_path: Path, province: str, start_date: Optional[str] = None) -> pd.DataFrame:
    with (data_path/filename(province)).open() as fp:
        top_level = json.load(fp)
    df = pd.DataFrame([(_[date], _[total_cases]) for _ in top_level[timeseries]], columns=["date", "total_cases"])
    df["date"] = (date_scale * df["date"]).apply(pd.Timestamp)
    df.set_index("date", inplace = True)
    if start_date:
        return df[df.index >= start_date]
    return df 

dkij_public, sulsel_public = [load_province_timeseries(Path("data"), _) for _ in provinces]


# private data: jakarta 
dkij_drop_cols = [
    'age', 'sex', 'fever', 'temp', 'cough', 'flu', 'sore_throat', 'shortness_breath', 'shivering', 'headache', 'malaise', 'muscle_pain',
    'nausea_vomiting', 'abdominal_pain', 'diarrhoea', 'date_recovered',
    'date_died', 'heart_disease', 'diabetes', 'pneumonia', 'hypertension', 'malignant',
    'immunology_disorder', 'chronic_kidney', 'chronic_liver', 'copd',
    'obesity', 'pregnant', 'tracing', 'otg', 'icu', 'intubation', 'ecmo',
    'criteria_cases', 'age_group', 'age_group2', 'date_discharge',
    'patient_status', 'death'
]

dkij = pd.read_stata("data/dkijakarta_180820.dta")\
         .query("province == 'DKI JAKARTA'")\
         .drop(columns=dkij_drop_cols + ["province"])
dkij["district"] = dkij.district.str.title()

dkij = dkij.groupby("date_positiveresult")["id"].count().rename("cases")

# private data: sulsel

schema = { 
    "Date Symptom Onset"          : "symptom_onset",
    "Date of Hospital  Admissions": "admission",
    "Date tested"                 : "tested",
    "Date of positive test result": "confirmed",
    "Date Recovered"              : "recovered",
    "Date Died"                   : "died",
    "Kebupaten/Kota"              : "regency",
    "Kecamatan"                   : "district"
}
def parse_datetimes(df):
    valid_idx = ~df.isna() & df.str.endswith("20")
    valid = df[valid_idx]
    monthfirst_idx = valid.str.endswith("/20") # short years -> month first notation 
    valid.loc[( monthfirst_idx)] = pd.to_datetime(valid[( monthfirst_idx)], errors = 'coerce', format = "%m/%d/%y", dayfirst = False)
    valid.loc[(~monthfirst_idx)] = pd.to_datetime(valid[(~monthfirst_idx)], errors = 'coerce', format = "%d/%m/%Y", dayfirst = True)
    # assert df.max() <= pd.to_datetime("October 03, 2020"), "date parsing resulted in future dates"
    df.loc[valid_idx] = valid.apply(pd.Timestamp)

sulsel = pd.read_csv("data/3 OCT 2020 Data collection template update South Sulawesi_CASE.csv", usecols = schema.keys())\
        .rename(columns = schema)\
        .dropna(how = 'all')
parse_datetimes(sulsel.loc[:, "confirmed"])

sulsel = sulsel.confirmed.value_counts().sort_index()

plt.plot(dkij.index, dkij.values, color = "royalblue", label = "private")
plt.plot(dkij_public.diff(), color = "firebrick", label = "public")
plt.legend()
plt.PlotDevice()\
    .title("\nJakarta: public vs private case counts")\
    .xlabel("date")\
    .ylabel("cases")
plt.xlim(right = dkij.index.max())
plt.ylim(top = 800)
plt.show()

plt.plot(sulsel,               color = "royalblue", label = "private", linewidth = 3)
plt.plot(sulsel_public.diff(), color = "firebrick", label = "public")
plt.legend()
plt.PlotDevice()\
    .title("\nSouth Sulawesi: public vs private case counts")\
    .xlabel("date")\
    .ylabel("cases")
plt.xlim(right = sulsel.index.max())
plt.show()

# correlation: dkij
dkij_diff = dkij_public.diff().dropna()
dkij_pub_clipped = dkij_diff[dkij_diff.index <= dkij.index.max()].iloc[1:]
dkij_idx = pd.date_range(dkij.index.min(), dkij.index.max())
dkij = dkij.reindex(dkij_idx, fill_value = 0)
dkij_corr = np.corrcoef(np.squeeze(dkij_pub_clipped.values), dkij.values)[1][0]

# correlation: sulsel
sulsel_idx = pd.date_range(sulsel.index.min(), sulsel.index.max())
sulsel = sulsel.reindex(sulsel_idx, fill_value = 0)

sulsel_diff = sulsel_public.diff()
sulsel_clipped = sulsel_diff[(sulsel_diff.index >= sulsel_idx.min()) & (sulsel_diff.index <= sulsel_idx.max())].fillna(0)
sulsel_corr = np.corrcoef(np.squeeze(sulsel_clipped.values), sulsel.values)[1][0]
