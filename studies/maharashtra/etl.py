from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

districts = [
	'AHMEDNAGAR','AKOLA','AMARAVATI','AURANGABAD',
	'BEED','BHANDARA','BULDHANA','CHANDRAPUR','DHULE',
	'GADCHIROLI','GONDIA','HINGOLI','JALGAON','JALNA',
	'KOLHAPUR','LATUR','MUMBAI','NAGPUR','NANDED','NANDURBAR',
 	'NASHIK','OSMANABAD','PALGHAR','PARBHANI','PUNE',
 	'RAIGAD','RATNAGIRI','SANGLI','SATARA','SINDHUDURG',
 	'SOLAPUR','THANE','WARDHA','WASHIM','YAVATMAL']

census_districts = [
	'AHMADNAGAR','AKOLA','AMRAVATI','AURANGABAD',
 	'BHANDARA','BID','BULDANA','CHANDRAPUR','DHULE',
 	'GADCHIROLI','GONDIYA','HINGOLI','JALGAON','JALNA',
 	'KOLHAPUR','LATUR','MUMBAI','MUMBAI (SUBURBAN)','NAGPUR','NANDED','NANDURBAR',
 	'NASHIK','OSMANABAD','PARBHANI','PUNE',
 	'RAIGARH','RATNAGIRI','SANGLI','SATARA','SINDHUDURG',
 	'SOLAPUR','THANE','WARDHA','WASHIM','YAVATMAL']

replacements = {
    "AHMADNAGAR": "AHMEDNAGAR", 
    "AMRAVATI"  : "AMARAVATI", 
    "BID"       : "BEED", 
    "BULDANA"   : "BULDHANA",
    "GONDIYA"   : "GONDIA", 
    "RAIGARH"   : "RAIGAD",
}

new_districts = ['PALGHAR']

def load_cases(path: Path) -> pd.DataFrame:
    cases = pd.read_csv(path, parse_dates=['Date'], dayfirst=True)
    cases['District'] = cases['District'].str.upper()
    return cases

def split_cases_by_district(cases: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {district: cases[cases["District"] == district] for district in districts}

def get_time_series(cases: pd.DataFrame) -> pd.Series:
    cases.set_index('Date', inplace=True)
    recovered = cases['daily_new_recoveries'] 
    infected  = cases['daily_new_cases']
    deceased  = cases['daily_new_deaths']

    time_series = pd.DataFrame({"Infected": infected, "Recovered": recovered, "Deceased": deceased}).fillna(0)
    return time_series

def assume_missing_0(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0

def log_delta(time_series: pd.DataFrame) -> pd.DataFrame:
    time_series["cases"] = assume_missing_0(time_series, "Infected") - assume_missing_0(time_series, "Recovered") -  assume_missing_0(time_series, "Deceased")
    log_delta = pd.DataFrame(np.log(time_series.cases)).rename(columns = {"cases" : "logdelta"})
    log_delta["time"] = (log_delta.index - log_delta.index.min()).days
    return log_delta

def log_delta_smoothed(time_series: pd.DataFrame) -> pd.DataFrame:
    time_series["cases"] = assume_missing_0(time_series, "Infected") - assume_missing_0(time_series, "Recovered") -  assume_missing_0(time_series, "Deceased")
    time_series["cases"] = lowess(time_series["cases"], time_series.index)[:, 1] 
    log_delta = pd.DataFrame(np.log(time_series.cases)).rename(columns = {"cases" : "logdelta"})
    log_delta["time"] = (log_delta.index - log_delta.index.min()).days
    return log_delta

# def district_migration_matrix(matrix_path: Path) -> np.matrix:
#     mm = pd.read_csv(matrix_path)
#     mm_state = mm[(mm.D_StateCensus2001 == "MAHARASHTRA") & (mm.O_StateCensus2001 == "MAHARASHTRA")]
#     pivot    = mm_state.pivot(index = "D_DistrictCensus2001", columns = "O_DistrictCensus2001", values = "NSS_STMigrants").fillna(0)
#     M  = np.matrix(pivot)
#     Mn = M/M.sum(axis = 0)
#     Mn[np.isnan(Mn)] = 0

#     districts = [replacements.get(district, district) for district in pivot.index]

#     return (districts, mm_state.groupby("O_DistrictCensus2001")["O_Population_2011"].agg(lambda x: list(x)[0]).values, Mn)


# def migratory_influx_matrix(influx_path: Path, num_migrants: int, release_rate: float) -> Dict[str, float]:
#     source_actives = np.array([21468, 80, 539, 461, 1007, 5291, 1893, 5254, 1595, 377, 1797, 7438, 148])
#     source_pops    = np.array([112400000.0, 34520000.0, 61100000.0, 35190000.0, 84580000.0, 60440000.0, 68550000.0, 26500000.0, 27740000.0, 27760000.0, 199800000.0, 72150000.0, 1055000.0])

#     flux = pd.read_csv(influx_path).fillna(0)
#     flux = flux[flux.State != "Other States"] 
#     flux = flux[flux.State != "Grand Total"] 
#     flux = flux[[col for col in flux.columns if col not in {"State", "Unknown", "Grand Total"}]]
#     # flux.rename(columns = {"Muzaffarpur" : "Muzzafarpur"})

#     # influx proportions
#     proportions = (flux * (source_actives/source_pops)[:, None]).sum(axis = 0)
#     return {replacements.get(k.upper(), k.upper()): v for (k, v) in (num_migrants * release_rate * (proportions / proportions.sum())).astype(int).to_dict().items()}