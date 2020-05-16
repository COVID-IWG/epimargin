from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from adaptive.utils import assume_missing_0

districts = [
    'Adilabad',
    'Hyderabad',
    'Jagtial',
    'Jangaon',
    'Mulugu',
    'Jogulamba Gadwal',
    'Kamareddy',
    'Karimnagar',
    'Khammam',
    'Komaram Bheem',
    'Mahabubabad',
    'Mahabubnagar',
    'Mancherial',
    'Medak',
    'Medchal Malkajgiri',
    'Nagarkurnool',
    'Nalgonda',
    'Nirmal',
    'Nizamabad',
    'Peddapalli',
    'Rajanna Sircilla',
    'Ranga Reddy',
    'Sangareddy',
    'Siddipet',
    'Suryapet',
    'Vikarabad',
    'Wanaparthy',
    'Warangal Rural',
    'Warangal Urban',
    'Yadadri Bhuvanagiri',
    'Bhadradri Kothagudem',
    'Jayashankar Bhupalapally',
    'Narayanpet'
]

columns_v3 = v3 = [
    'Patient Number',
    'State Patient Number',
    'Date Announced',
    'Estimated Onset Date',
    'Age Bracket',
    'Gender',
    'Detected City',
    'Detected District',
    'Detected State',
    'State code',
    'Current Status',
    'Notes',
    'Contracted from which Patient (Suspected)',
    'Nationality',
    'Type of transmission',
    'Status Change Date',
    'Source_1',
    'Source_2',
    'Source_3',
    'Backup Notes',
    'Num cases'
]

drop_cols_v3 = {
    "Age Bracket",
    "Gender",
    "Detected City",
    "Notes",
    'Contracted from which Patient (Suspected)', 
    'Nationality',
    "Source_1",
    "Source_2",
    "Source_3",
    "Backup Notes",
    "State Patient Number",
    "State code",
    "Estimated Onset Date",
    "Type of transmission"
}

columns_v4 = v4 = [
    'Entry_ID', 
    'State Patient Number', 
    'Date Announced', 
    'Age Bracket',
    'Gender', 
    'Detected City', 
    'Detected District', 
    'Detected State',
    'State code', 
    'Num Cases', 
    'Current Status',
    'Contracted from which Patient (Suspected)', 
    'Notes', 
    'Source_1',
    'Source_2', 
    'Source_3', 
    'Nationality', 
    'Type of transmission',
    'Status Change Date', 
    'Patient Number'
]

drop_cols_v4 = {
    "Entry_ID",
    'Age Bracket',
    'Gender', 
    'Detected City',
    'State code',
    'Contracted from which Patient (Suspected)',
    'Notes', 
    'Source_1',
    'Source_2', 
    'Source_3', 
    'Nationality', 
    'Type of transmission',
    "State Patient Number"
}

column_ordering_v4  = [
    'Patient Number',
    'Date Announced',
    'Detected District',
    'Detected State',
    'Current Status',
    'Status Change Date',
    'Num cases'
]

def load_data_v3(path: Path):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v3) - drop_cols_v3,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    return cases[cases["Detected State"] == "Telangana"]

def load_data_v4(path: Path):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v4) - drop_cols_v4,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    cases["Num cases"] = cases["Num Cases"]
    return cases[cases["Detected State"] == "Telangana"][column_ordering_v4]

def get_time_series(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby(["Status Change Date", "Current Status"])["Patient Number"].count().unstack().fillna(0)
    totals["date"]     = totals.index
    totals["time"]     = (totals["date"] - totals["date"].min()).dt.days
    totals["logdelta"] = np.log(assume_missing_0(totals, "Hospitalized") - assume_missing_0(totals, "Recovered") -  assume_missing_0(totals, "Deceased"))
    return totals

def load_all_data(path1: Path, path2: Path, path3: Path) -> pd.DataFrame:
    cases1 = load_data_v3(path1)
    cases2 = load_data_v3(path2)
    cases3 = load_data_v4(path3)
    all_cases = pd.concat([cases1, cases2, cases3])
    all_cases["Status Change Date"] = all_cases["Status Change Date"].fillna(all_cases["Date Announced"])
    return all_cases

def district_migration_matrix(gdf_path: Path, population_path: Path) -> Tuple[Sequence[str], Sequence[float], np.matrix]:
    gdf = gpd.read_file(gdf_path)
    districts = list(gdf.district.values)

    pop_df = pd.read_csv(population_path)
    population_mapping = {k.replace("-", " "): float(v.replace(",", "")) for (k, v) in zip(pop_df["Name"], pop_df["Population(2011 census)"])}
    populations = [population_mapping[district] for district in districts]

    centroids = [list(pt.coords)[0] for pt in gdf.centroid]
    P = distance_matrix(centroids, centroids)
    P[P != 0] = P[P != 0] ** -1.0 
    P *= np.array(populations)[:, None]
    P /= P.sum(axis = 0)

    return (districts, populations, P)

def log_delta(time_series: pd.DataFrame) -> pd.DataFrame:
    time_series["cases"] = assume_missing_0(time_series, "Hospitalized") - assume_missing_0(time_series, "Recovered") -  assume_missing_0(time_series, "Deceased")
    log_delta = pd.DataFrame(np.log(time_series.cases)).rename(columns = {"cases" : "logdelta"})
    log_delta["time"] = (log_delta.index - log_delta.index.min()).days
    return log_delta

def split_cases_by_district(cases: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {district: cases[cases["Detected District"] == district] for district in districts}