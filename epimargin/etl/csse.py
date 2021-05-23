from pathlib import Path
from typing import Optional

import pandas as pd

from .commons import download_data

""" tools to download and load data from JHU's CSSE Covid tracker """

CSSE_REPO_BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"
DATE_FMT = "%m-%d-%Y"

DROP_SCHEMA_V1 = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incidence_Rate", "Case-Fatality_Ratio", "Country_Region"]
DROP_SCHEMA_V2 = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incident_Rate",  "Case_Fatality_Ratio", "Country_Region"]

def fetch(dst: Path, date: pd.Timestamp, overwrite: bool = False) -> None:
    filename = date.strftime(DATE_FMT) + ".csv"
    if (not (dst/filename).exists()) or overwrite:
        download_data(dst, filename, CSSE_REPO_BASE_URL)

def fetch_range(dst: Path, start: str, end: str):
    for date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end)):
        fetch(dst, date)

def load(dst: Path, start: str, end: str, selector: Optional[str] = None) -> pd.DataFrame:
    return pd.concat([
        (lambda _: _.query(selector) if selector else _)(pd.read_csv(dst/(date.strftime(DATE_FMT) + ".csv"))).assign(date = date) for date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end))
    ], axis = 0)

def load_country(dst: Path, start: str, end: str, country: str, schema_version: int = 1):
    return load(dst, start, end, f"Country_Region == '{country}'")\
        .drop(columns = DROP_SCHEMA_V1 if schema_version == 1 else DROP_SCHEMA_V2)\
        .assign(Active = lambda _:_["Active"].astype(int))

def assemble_timeseries(df: pd.DataFrame, province: Optional[str] = None):
    totals = (
        df[df.Province_State == province].set_index("date") if province else 
        df.set_index(["date", "Province_State"]).stack().sum(level = [0, 2]).unstack()
    )[["Deaths", "Recovered", "Confirmed"]]\
        .rename(columns = {"Confirmed": "T", "Deaths": "D", "Recovered": "R"})
    return pd.concat([
        totals, 
        totals.diff()\
            .rename(lambda x: "d" + x, axis = 1)
        ], axis = 1)\
            .dropna()\
            .astype(int)