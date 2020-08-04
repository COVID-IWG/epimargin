from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

state_name_lookup = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District Of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MP': 'Northern Mariana Islands',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VI': 'US Virgin Islands',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

colours = [
    "darkorange", "tomato", "olivedrab", 
    "forestgreen", "lightseagreen", "deepskyblue",
    "mediumpurple", "darkmagenta"
    ]

google_mobility_columns = [
    "sub_region_1",
    "sub_region_2",
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
 ]

def load_country_google_mobility(country_code: str) -> pd.DataFrame:
    full_df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', parse_dates = ["date"])
    country_df = full_df[full_df["country_region_code"] == country_code]
    country_df["sub_region_1"] = country_df["sub_region_1"].str.replace(" and ", " & ")
    return country_df[google_mobility_columns]

def load_us_county_data(file: str, url: Optional[str] = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/") -> pd.DataFrame:
    df = pd.read_csv(url + file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["state_name"] = df["state"].map(state_name_lookup)
    return df

def load_intervention_data() -> pd.DataFrame:
    interventions = pd.read_csv('https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/raw_data/national/public_implementations_fips.csv')
    interventions.rename(columns={"Unnamed: 1": "county_name", "Unnamed: 2": "state"}, inplace=True)
    interventions["county_name"] = interventions["county_name"].str.title()
    interventions = pd.DataFrame(interventions.iloc[:,1:].set_index(["state","county_name"]).stack(), columns=["date"]).reset_index().rename(columns={"level_2": "intervention"})
    interventions["state_name"] = interventions["state"].map(state_name_lookup)
    interventions["date"] = pd.to_datetime(interventions["date"]+ "-2020", format="%d-%b-%Y")
    interventions = interventions.set_index(["state_name", "county_name", "date"]).iloc[:,1:].sort_index()
    return pd.get_dummies(interventions)

def load_rt_estimations(data_path: Path) -> pd.DataFrame:
    rt_df = pd.read_csv(data_path, parse_dates = ["date"])
    rt_df["state_name"] = rt_df["state"].map(state_name_lookup)
    return rt_df.iloc[:, 1:].set_index(["state_name", "date"])

def load_metro_areas(data_path: Path) -> pd.DataFrame:
    metro_df = pd.read_csv(data_path)
    return metro_df[metro_df["area_type"] == "Metro"][["cbsa_fips", "county_fips", "county_name", "state_codes", "state_name"]]

def get_case_timeseries(case_df: pd.DataFrame) -> pd.DataFrame:
    county_cases = pd.DataFrame(case_df.set_index(["state_name", "county_name"]).iloc[:,3:].stack()).rename(columns={0:"cumulative_confirmed_cases"}).reset_index()
    county_cases["date"] = pd.to_datetime(county_cases["level_2"],format="%m/%d/%y")
    county_cases = county_cases.set_index(["state_name", "county_name", "date"]).iloc[:,1:]
    county_cases["cumulative_confirmed_cases"] = pd.to_numeric(county_cases["cumulative_confirmed_cases"])
    county_cases = county_cases.groupby(["state_name", "county_name"]).apply(add_delta_col)
    county_cases["daily_confirmed_cases"].fillna(county_cases["cumulative_confirmed_cases"], inplace=True)
    return county_cases["daily_confirmed_cases"]

def add_lag_cols(grp: pd.DataFrame):
    for lag in [1, 7, 14]:
        for col in ["daily_confirmed_cases","retail_and_recreation_percent_change_from_baseline", "grocery_and_pharmacy_percent_change_from_baseline", "parks_percent_change_from_baseline", "transit_stations_percent_change_from_baseline", "workplaces_percent_change_from_baseline", "residential_percent_change_from_baseline"]:
            grp[col + '_lag_' + str(lag)] = grp[col].shift(-lag)
    return grp

def pop_prop_col(grp):
    grp['pop_prop'] = grp['population'] / grp['population'].sum()
    return grp

def add_delta_col(grp: pd.DataFrame):
    grp["daily_confirmed_cases"] = grp["cumulative_confirmed_cases"].diff()
    return grp

def add_colours(intervention_df):
    dic = {}
    for i, k in enumerate(list(intervention_df["intervention"].unique())):
        dic[k] = colours[i]
    intervention_df["colour"] = intervention_df["intervention"].map(dic)
    return intervention_df



