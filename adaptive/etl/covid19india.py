from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from adaptive.utils import assume_missing_0

"""code to extract time series for COVID19India.org data"""

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

state_replacements = {
    'Dadra & Nagar Haveli': 'Dadra & Nagar Haveli & Daman & Diu',
    'Daman & Diu': 'Dadra & Nagar Haveli & Daman & Diu'
}
 
def data_path(i: int):
    return f"raw_data{i}.csv"

def download_data(data_path: Path, filename: str, base_url: str = 'https://api.covid19india.org/csv/latest/'):
    url = base_url + filename
    response = requests.get(url)
    (data_path/filename).open('wb').write(response.content)

def standardize_column_headers(df: pd.DataFrame):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ","_").str.replace('[^a-zA-Z0-9_]', '')

# load data until April 26
def load_data_v3(path: Path):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v3) - drop_cols_v3,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    standardize_column_headers(cases)
    return cases

# load data for April 27 - May 09  
def load_data_v4(path: Path):
    cases = pd.read_csv(path, 
        usecols     = set(columns_v4) - drop_cols_v4,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["Date Announced", "Status Change Date"])
    standardize_column_headers(cases)
    return cases[column_ordering_v4]

def add_time_col(grp_df):
    grp_df['time'] = (grp_df["date"] - grp_df["date"].min()).dt.days
    return grp_df

# calculate daily totals and growth rate
def get_time_series(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    totals = df.groupby(group_cols).sum()
    if len(totals) == 0:
        return pd.DataFrame()
    totals["date"] = totals.index.get_level_values('status_change_date')
    totals = totals.groupby(level=0).apply(add_time_col) if len(group_cols) > 1 else add_time_col(totals)
    totals["delta"] = assume_missing_0(totals, "Hospitalized") - assume_missing_0(totals, "Recovered") - assume_missing_0(totals, "Deceased")
    totals["logdelta"] = np.ma.log(totals["delta"].values).filled(0)
    return totals

def load_all_data(v3_paths: Sequence[Path], v4_paths: Sequence[Path]) -> pd.DataFrame:
    cases_v3 = [load_data_v3(path) for path in v3_paths]
    cases_v4 = [load_data_v4(path) for path in v4_paths]
    all_cases = pd.concat(cases_v3 + cases_v4)
    all_cases["status_change_date"] = all_cases["status_change_date"].fillna(all_cases["date_announced"])
    for col in ["detected_state", "detected_district"]:
        all_cases[col] = all_cases[col].str.strip().str.title().str.replace(' And ', ' & ').str.replace('  ', ' ')
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

def redistribute_missing_cases(
    df_cases: pd.DataFrame,
    current_districts: pd.DataFrame,
    new_states: Sequence[str],
    populations: pd.DataFrame,
    fraction: bool = True) -> pd.DataFrame:
    totals = df_cases.groupby(['detected_state','detected_district','status_change_date','current_status'])["num_cases"].agg(lambda counts: np.sum(np.abs(counts)))
    totals = totals.unstack().fillna(0)[['Deceased','Hospitalized','Recovered']].reset_index()   
    mask = totals['detected_state'].isin(new_states) | ((totals['detected_state'].isin(current_districts['state'])) & (totals['detected_district'].isin(current_districts['district'])))
    missing_cases = totals[~mask].groupby(['detected_state','status_change_date']).sum()
    actual_cases = totals[mask]
    populations = populations.groupby(level=0).apply(add_pop_fraction)
    additions = get_additional_cases(missing_cases, populations, fraction)
    additions['detected_district'] = additions['district_name']
    return pd.concat([additions.iloc[:,:-1], actual_cases]).set_index(['detected_state','detected_district','status_change_date'])

def add_pop_fraction(grp):
    grp['population_fraction'] = grp['population'] / grp['population'].sum()
    return grp

def get_additional_cases(missing_cases: pd.DataFrame, populations: pd.DataFrame, fraction: bool):
    additions = pd.DataFrame(columns=['detected_state','status_change_date','detected_district','Deceased', 'Hospitalized', 'Recovered'])
    for state in missing_cases.groupby(level=0):
        for date in state[1].index.get_level_values(level=1):
            if state[0] in populations.index.get_level_values(level=0):
                df = populations.xs(state[0], level=0).groupby(level=0).apply(add_missing_cases, missing_cases.loc[state[0],date], fraction).reset_index()[['district_name', 'Deceased','Hospitalized','Recovered']]
                df['detected_state'], df['status_change_date'] = state[0], date
                additions = additions.append(df)
    return additions

def add_missing_cases(grp, missing, fraction):
    for col in ['Deceased','Hospitalized','Recovered']:
        grp[col] = grp['population_fraction'] * missing[col]
    return grp

def get_current_state_districts(
    migration_data: pd.DataFrame,
    district_matches: pd.DataFrame) -> pd.DataFrame:
    current_state_districts = pd.concat([district_matches[['state', 'district_2011']].rename(columns={'district_2011': 'district'}),
                                         migration_data[['O_StateCensus2011','O_DistrictCensus2011']].rename(columns={'O_StateCensus2011': 'state', 'O_DistrictCensus2011': 'district'})])
    return current_state_districts.drop_duplicates().dropna()

def load_populations(pop_path: Path):
    pops = pd.read_csv(pop_path)
    pops.columns = pops.columns.str.lower().str.replace(' ', '_')
    for col in ['state_name', 'district_name']:
        pops[col] = pops[col].str.title().str.strip().str.replace('  ', ' ').str.replace(' And ', ' & ')
    pops['population'] = pd.to_numeric(pops['population'].str.replace(',', ''))
    return pops.set_index(['state_name', 'district_name'])

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M 

def load_migration_data(matrix_path: Path):
    mm = pd.read_csv(matrix_path)
    for col in  ['D_StateCensus2011', 'D_DistrictCensus2011', 'O_StateCensus2011', 'O_DistrictCensus2011']:
        mm[col] = mm[col].str.title().str.replace('  ', ' ')
    return mm.replace(state_replacements)

def district_migration_matrices(
    mm: pd.DataFrame, 
    states: Sequence[str]) -> Dict[str, np.matrix]:
    aggregations = dict()
    for state in  states:
        mm_state = mm[(mm.D_StateCensus2011 == state) & (mm.O_StateCensus2011 == state)]
        # handle states that need migration data combined (e.g. Mumbai and Mumbai Suburban)
        if state in district_2011_replacements:
            mm_state.replace(district_2011_replacements[state], inplace=True)
        # group to combine multiple districts with same name based on above
        grouped_mm_state = mm_state.groupby(['D_DistrictCensus2011', 'O_DistrictCensus2011'])[['O_Population_2011','NSS_STMigrants']].sum().reset_index()
        pivot    = grouped_mm_state.pivot(index = "D_DistrictCensus2011", columns = "O_DistrictCensus2011", values = "NSS_STMigrants").fillna(0)
        M  = np.matrix(pivot)
        Mn = M/M.sum(axis = 0)
        Mn[np.isnan(Mn)] = 0
        aggregations[state] = (
            pivot.index, 
            grouped_mm_state.groupby("O_DistrictCensus2011")["O_Population_2011"].agg(lambda x: list(x)[0]).values, 
            Mn
        )
    return aggregations 
