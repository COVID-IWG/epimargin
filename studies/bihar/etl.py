from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from adaptive.estimators import lowess

collect = 'DATE OF 1st SAMPLE COLLECTION'
confirm = 'DATE OF POSITIVE TEST CONFIRMATION'
release = 'DATE OF DISCHARGE'
death   = "DATE OF DEATH"

columns = [
    "SL NO.",
    "CASE ID",
    "AGE",
    "GENDER",
    "DISTRICT",
    "BLOCK",
    "CAUSE OF SAMPLE COLLECTION ",
    "CATEGORY",
    "PRESENT STATUS",
    "1ST TEST (POSITIVE ) TESTING LAB",
    "DATE OF 1st SAMPLE COLLECTION",
    "DATE OF POSITIVE TEST CONFIRMATION",
    "DATE OF DISCHARGE",
    "SYMPTOMS",
    "DATE OF DEATH",
    "CASE STATUS"
]

drop_cols = set([
    "SL NO.", "CASE ID", "AGE", "GENDER", "BLOCK", "CAUSE OF SAMPLE COLLECTION ", 
    "CATEGORY", "PRESENT STATUS", "1ST TEST (POSITIVE ) TESTING LAB", "SYMPTOMS"
])

districts = [
    'MUNGER', 'PATNA', 'SIWAN', 'NALANDA', 'LAKHISARAI', 'GOPALGANJ',
    'GAYA', 'BEGUSARAI', 'SARAN', 'BHAGALPUR', 'NAWADA', 'VAISHALI',
    'BUXAR', 'BHOJPUR', 'ROHTAS', 'EAST CHAMPARAN', 'BANKA', 'KAIMUR',
    'MADHEPURA', 'AURANGABAD', 'ARWAL', 'JEHANABAD', 'MADHUBANI',
    'PURNEA', 'DARBHANGA', 'ARARIA', 'SHEIKPURA', 'SITAMARHI',
    'WEST CHAMPARAN', 'KATIHAR', 'SHEOHAR', 'SAMASTIPUR', 'KISHANGANJ',
    'SUPAUL', 'SAHARSA', 'KHAGARIA', 'MUZZAFARPUR'
]

replacements = {
    "PASHCHIM CHAMPARAN": "WEST CHAMPARAN", 
    "PURBA CHAMPARAN"   : "EAST CHAMPARAN", 
    "KAIMUR (BHABUA)"   : "KAIMUR", 
    "MUZAFFARPUR"       : "MUZZAFARPUR", 
    "SHEIKHPURA"        : "SHEIKPURA",
    "SHIEKHPURA"        : "SHEIKPURA",
    "PURNIA"            : "PURNEA"
}

def load_cases(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[collect, confirm, release, death], dayfirst=True, usecols=lambda col: col not in drop_cols)

def split_cases_by_district(cases: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {district: cases[cases["DISTRICT"] == district] for district in districts}

def get_time_series(cases: pd.DataFrame) -> pd.Series:
    I = cases["CASE STATUS"].isna()
    R = cases["CASE STATUS"] == "RECOVERED"
    D = cases["CASE STATUS"] == "DECEASED"

    infected  = cases[I].groupby(["DISTRICT", confirm])[confirm].count()\
                        .rename_axis(index = {confirm: "time"})\
                        .rename("infected")
    recovered = cases[R].groupby(["DISTRICT", release])["CASE STATUS"].count()\
                        .rename_axis(index = {release: "time"})\
                        .rename("recovered")
    deceased  = cases[D].groupby(["DISTRICT", death]   )["CASE STATUS"].count()\
                        .rename_axis(index = {death:   "time"})\
                        .rename("deceased")

    time_series = pd.DataFrame({"infected": infected, "recovered": recovered, "deceased": deceased}).fillna(0)
    time_series.unstack().fillna(0).stack().apply(lambda ts: ts.infected - ts.recovered - ts.deceased, axis = 1).rename("cases")
    return time_series

def district_migration_matrix(matrix_path: Path) -> np.matrix:
    mm = pd.read_csv(matrix_path)
    mm_state = mm[(mm.D_StateCensus2001 == "BIHAR") & (mm.O_StateCensus2001 == "BIHAR")]
    pivot    = mm_state.pivot(index = "D_DistrictCensus2001", columns = "O_DistrictCensus2001", values = "NSS_STMigrants").fillna(0)
    M  = np.matrix(pivot)
    Mn = M/M.sum(axis = 0)
    Mn[np.isnan(Mn)] = 0

    districts = [replacements.get(district, district) for district in pivot.index]

    return (districts, mm_state.groupby("O_DistrictCensus2001")["O_Population_2011"].agg(lambda x: list(x)[0]).values, Mn)

def migratory_influx_matrix(influx_path: Path, num_migrants: int, release_rate: float) -> Dict[str, float]:
    # sources = ['Maharashtra', 'Kerala', 'Karnataka', 'Telangana', 'Andhra Pradesh', 'Gujarat', 'Rajasthan', 'Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Tamil Nadu', 'Chandigarh']
    source_actives = np.array([21468, 80, 539, 461, 1007, 5291, 1893, 5254, 1595, 377, 1797, 7438, 148])
    source_pops    = np.array([112400000.0, 34520000.0, 61100000.0, 35190000.0, 84580000.0, 60440000.0, 68550000.0, 26500000.0, 27740000.0, 27760000.0, 199800000.0, 72150000.0, 1055000.0])

    flux = pd.read_csv(influx_path).fillna(0)
    flux = flux[flux.State != "Other States"] 
    flux = flux[flux.State != "Grand Total"] 
    flux = flux[[col for col in flux.columns if col not in {"State", "Unknown", "Grand Total"}]]

    # influx proportions
    proportions = (flux * (source_actives/source_pops)[:, None]).sum(axis = 0)
    return {replacements.get(k.upper(), k.upper()): v for (k, v) in (num_migrants * release_rate * (proportions / proportions.sum())).astype(int).to_dict().items()}
