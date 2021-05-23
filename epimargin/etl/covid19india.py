from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..utils import assume_missing_0

""" tools to extract time series for COVID19India.org data """

# states created after the 2001 census
new_states = set("Telangana")

# states renamed in 2011
renamed_states = { 
    "Orissa"      : "Odisha",
    "Pondicherry" : "Puducherry"
}

columns_v1 = v1 = [
    "patient number",
    "state patient number",
    "date announced",
    "age bracket",
    "gender",
    "detected city",
    "detected district",
    "detected state",
    "current status",
    "notes",
    "contracted from which patient (suspected)",
    "nationality",
    "type of transmission",
    "status change date",
    "source_1",
    "source_2",
    "source_3",
    "backup note"
]

columns_v2 = v2 = [
    'patient number',
    'state patient number',
    'date announced',
    'estimated onset date',
    'age bracket',
    'gender',
    'detected city',
    'detected district',
    'detected state',
    'state code',
    'current status',
    'notes',
    'contracted from which patient (suspected)',
    'nationality',
    'type of transmission',
    'status change date',
    'source_1',
    'source_2',
    'source_3',
    'backup notes'
]

drop_cols = {
    "age bracket",
    "gender",
    "detected city",
    # "detected district",
    "notes",
    "contracted from which patient (suspected)",
    "nationality",
    "source_1",
    "source_2",
    "source_3",
    "backup note",
    "backup notes",
    "type of transmission"
}

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
    'Num Cases'
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
    'patient_number',
    'date_announced',
    'detected_district',
    'detected_state',
    'current_status',
    'status_change_date',
    'num_cases'
 ]

district_2011_replacements = {
    'Maharashtra' : {
        'Mumbai Suburban' : 'Mumbai'}
 }

state_code_lookup = {
    'AN'  : 'Andaman & Nicobar Islands',
    'AP'  : 'Andhra Pradesh',
    'AR'  : 'Arunachal Pradesh',
    'AS'  : 'Assam',
    'BR'  : 'Bihar',
    'CH'  : 'Chandigarh',
    'CT'  : 'Chhattisgarh',
    'DD'  : 'Daman & Diu', 
    'DNDD': 'Dadra & Nagar Haveli and Daman & Diu',
    'DL'  : 'Delhi',
    'DN'  : 'Dadra & Nagar Haveli',
    'GA'  : 'Goa',
    'GJ'  : 'Gujarat',
    'HP'  : 'Himachal Pradesh',
    'HR'  : 'Haryana',
    'JH'  : 'Jharkhand',
    'JK'  : 'Jammu & Kashmir',
    'KA'  : 'Karnataka',
    'KL'  : 'Kerala',
    'LA'  : 'Ladakh',
    'LD'  : 'Lakshadweep',
    'MH'  : 'Maharashtra',
    'ML'  : 'Meghalaya',
    'MN'  : 'Manipur',
    'MP'  : 'Madhya Pradesh',
    'MZ'  : 'Mizoram',
    'NL'  : 'Nagaland',
    'OR'  : 'Odisha',
    'PB'  : 'Punjab',
    'PY'  : 'Puducherry',
    'RJ'  : 'Rajasthan',
    'SK'  : 'Sikkim',
    'TG'  : 'Telangana',
    'TN'  : 'Tamil Nadu',
    'TR'  : 'Tripura',
    'TT'  : 'India',
    'UN'  : 'State Unassigned',
    'UP'  : 'Uttar Pradesh',
    'UT'  : 'Uttarakhand',
    'WB'  : 'West Bengal',
}

state_name_lookup = { 
    'Andaman & Nicobar Islands'  : 'AN', 
    'Andaman And Nicobar Islands': 'AN', 
    'Andaman and Nicobar Islands': 'AN', 
    'Andhra Pradesh'             : 'AP', 
    'Arunachal Pradesh'          : 'AR', 
    'Assam'                      : 'AS', 
    'Bihar'                      : 'BR', 
    'Chandigarh'                 : 'CH', 
    'Chhattisgarh'               : 'CT', 
    'Daman & Diu'                : 'DD', 
    'Daman And Diu'              : 'DD', 
    'Daman and Diu'              : 'DD', 
    'Delhi'                      : 'DL', 
    'Dadra & Nagar Haveli'       : 'DN', 
    'Dadra And Nagar Haveli'     : 'DN', 
    'Dadra and Nagar Haveli'     : 'DN', 
    'Goa'                        : 'GA', 
    'Gujarat'                    : 'GJ', 
    'Himachal Pradesh'           : 'HP', 
    'Haryana'                    : 'HR', 
    'Jharkhand'                  : 'JH', 
    'Jammu & Kashmir'            : 'JK', 
    'Jammu and Kashmir'          : 'JK', 
    'Jammu And Kashmir'          : 'JK', 
    'Karnataka'                  : 'KA', 
    'Kerala'                     : 'KL', 
    'Ladakh'                     : 'LA', 
    'Lakshadweep'                : 'LD', 
    'Maharashtra'                : 'MH', 
    'Meghalaya'                  : 'ML', 
    'Manipur'                    : 'MN', 
    'Madhya Pradesh'             : 'MP', 
    'Mizoram'                    : 'MZ', 
    'Nagaland'                   : 'NL', 
    'Odisha'                     : 'OR', 
    'Punjab'                     : 'PB', 
    'Puducherry'                 : 'PY', 
    'Rajasthan'                  : 'RJ', 
    'Sikkim'                     : 'SK', 
    'Telangana'                  : 'TG', 
    'Tamil Nadu'                 : 'TN', 
    'Tripura'                    : 'TR', 
    'India'                      : 'TT', 
    'State Unassigned'           : 'UN', 
    'Uttar Pradesh'              : 'UP', 
    'Uttarakhand'                : 'UT',
    'West Bengal'                : 'WB',
    'Dadra And Nagar Haveli And Daman And Diu' : "DNDD"
}


def data_path(i: int):
    return f"raw_data{i}.csv"

def standardize_column_headers(df: pd.DataFrame):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ","_").str.replace('[^a-zA-Z0-9_]', '')

# load data until April 26
def load_data_v3(path: Path, drop = drop_cols_v3):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v3) - drop,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    standardize_column_headers(cases)
    return cases

# load data for April 27 - May 09  
def load_data_v4(path: Path, drop = drop_cols_v3):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v4) - drop,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    standardize_column_headers(cases)
    return cases[column_ordering_v4]

def add_time_col(grp_df):
    grp_df['time'] = (grp_df["date"] - grp_df["date"].min()).dt.days
    return grp_df

# calculate daily totals and growth rate
def get_time_series(df: pd.DataFrame, group_col: Optional[Sequence[str]] = None, drop_negatives = True) -> pd.DataFrame:
    if group_col:
        group_cols = (group_col if isinstance(group_col, list) else [group_col]) + ["status_change_date", "current_status"]
    else: 
        group_cols = ["status_change_date", "current_status"]
    if drop_negatives:
        df = df[df["num_cases"] >= 0]
    totals = df.groupby(group_cols)["num_cases"].agg(lambda counts: np.sum(np.abs(counts)))
    if len(totals) == 0:
        return pd.DataFrame()
    totals = totals.unstack().fillna(0)[['Deceased','Hospitalized','Recovered']]
    totals["date"] = totals.index.get_level_values("status_change_date")
    totals = totals.groupby(level=0).apply(add_time_col) if group_col else add_time_col(totals)
    totals["delta"] = assume_missing_0(totals, "Hospitalized") - assume_missing_0(totals, "Recovered") - assume_missing_0(totals, "Deceased")
    totals["logdelta"] = np.ma.log(totals["delta"].values).filled(0)
    return totals

def load_all_data(v3_paths: Sequence[Path], v4_paths: Sequence[Path]) -> pd.DataFrame:
    cases_v3 = [load_data_v3(path) for path in v3_paths]
    cases_v4 = [load_data_v4(path) for path in v4_paths]
    all_cases = pd.concat(cases_v3 + cases_v4)
    all_cases["status_change_date"] = all_cases["status_change_date"].fillna(all_cases["date_announced"])
    all_cases["detected_state"]     = all_cases["detected_state"].str.strip().str.title()
    all_cases["detected_district"]  = all_cases["detected_district"].str.strip().str.title()  
    return all_cases.dropna(subset  = ["detected_state"])

# assuming analysis for data structure from COVID19-India saved as resaved, properly-quoted file (v1 and v2)
def load_data(datapath: Path, reduced: bool = False, schema: Optional[Sequence[str]] = None) -> pd.DataFrame: 
    if not schema:
        schema = columns_v1
    df = pd.read_csv(datapath, 
        skiprows    = 1, # supply fixed header in order to deal with Google Sheets export issues 
        names       = schema, 
        usecols     = (lambda _: _ not in drop_cols) if reduced else None,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    standardize_column_headers(df)
    return df

# replace covid api detected_district names with 2011 district name
def replace_district_names(df_state: pd.DataFrame, state_district_maps: pd.DataFrame) -> pd.DataFrame:
    state_district_maps = state_district_maps[['district_covid_api', 'district_2011']].set_index('district_covid_api')
    district_map_dict = state_district_maps.to_dict()['district_2011']
    df_state['detected_district'].replace(district_map_dict, inplace=True)
    return df_state

def load_statewise_data(statewise_data_path: Path, drop_unassigned: bool = True) -> pd.DataFrame:
    df_raw = pd.read_csv(statewise_data_path, parse_dates = ["Date"])
    df_raw.rename(columns=state_code_lookup, inplace=True)
    df = pd.DataFrame(df_raw.set_index(["Date","Status"]).unstack().unstack()).reset_index()
    df.columns = ["state", "current_status", "status_change_date", "num_cases"]
    df.replace("Confirmed", "Hospitalized", inplace=True)
    # drop negative cases and cases with no state assigned
    df_assigned = df[(~df["state"].isin(["Date_YMD", "State Unassigned", "TT"] if drop_unassigned else ["Date_YMD", "TT"]))]
    return df_assigned[(df_assigned["num_cases"] >= 0)]
