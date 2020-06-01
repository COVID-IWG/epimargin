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
    time_series = cases.groupby('Date')[['daily_new_cases','daily_new_deaths','daily_new_recoveries']].agg(lambda counts: np.sum(np.abs(counts)))

    return time_series.rename(columns={"daily_new_cases" : "Infected", "daily_new_recoveries" : "Recovered", "daily_new_deaths" : "Deceased"}).fillna(0)

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
