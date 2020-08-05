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

state_interventions_info = {
 'STATE': 'State',
 'POSTCODE': 'State Abbreviation',
 'STEMERG': 'State of emergency',
 'CLSCHOOL': 'Date closed K-12 schools',
 'CLDAYCR': 'Closed day cares',
 'OPNCLDCR': 'Reopen day cares',
 'CLNURSHM': 'Date banned visitors to nursing homes',
 'STAYHOME': 'Stay at home/ shelter in place',
 'END_STHM': 'End/relax stay at home/shelter in place',
 'CLBSNS': 'Closed non-essential businesses',
 'END_BSNS': 'Began to reopen businesses',
 'RELIGEX': 'Religious Gatherings Exempt Without Clear Social Distance Mandate*',
 'FM_ALL': 'Mandate face mask use by all individuals in public spaces',
 'FMFINE': 'Face mask mandate enforced by fines',
 'FMCITE': 'Face mask mandate enforced by criminal charge/citation',
 'FMNOENF': 'No legal enforcement of face mask mandate',
 'FM_EMP': 'Mandate face mask use by employees in public-facing businesses',
 'ALCOPEN': 'Alcohol/Liquor Stores Open',
 'ALCREST': 'Allow restaurants to sell takeout alcohol',
 'ALCDELIV': 'Allow restaurants to deliver alcohol',
 'GUNOPEN': 'Keep Firearms Sellers Open',
 'CLREST': 'Closed restaurants except take out',
 'ENDREST': 'Reopen restaurants',
 'RSTOUTDR': 'Initially reopen restaurants for outdoor dining only',
 'CLGYM': 'Closed gyms',
 'ENDGYM': 'Reopened gyms',
 'CLMOVIE': 'Closed movie theaters',
 'END_MOV': 'Reopened movie theaters',
 'CLOSEBAR': 'Closed Bars',
 'END_BRS': 'Reopen bars',
 'END_HAIR': 'Reopened hair salons/barber shops',
 'END_CONST': 'Restart non-essential construction',
 'END_RELG': 'Reopen Religious Gatherings',
 'ENDRETL': 'Reopen non-essential retail',
 'BCLBAR2': 'Begin to Re-Close Bars',
 'CLBAR2': 'Re-Close Bars (statewide)',
 'CLMV2': 'Re-Close Movie Theaters (statewide)',
 'CLGYM2': 'Re-Close Gyms (statewide)',
 'CLRST2': 'Re-Close Indoor Dining (Statewide)',
 'QRSOMEST': 'Mandate quarantine for those entering the state from specific states',
 'QR_ALLST': 'Mandate quarantine for all individuals entering the state from another state',
 'QR_END': 'Date all mandated quarantines ended'
 }
 
state_meta_data = {
 'FIPS': 'FIPS Code',
 'POPDEN18': 'Population density per square miles',
 'POP18': 'Population 2018 ',
 'SQML': 'Square Miles',
 'HMLS19': 'Number Homeless (2019)',
 'UNEMP18': 'Percent Unemployed (2018). ',
 'POV18': 'Percent living under the federal poverty line (2018). ',
 'RISKCOV': 'Percent at risk for serious illness due to COVID',
 'DEATH18': 'All-cause deaths 2018'
 }

county_mask_cols = [
 'state_fips', 
 'state_name',
 'county_fips',
 'county_name',
 'county_mask_policy_start',
 'state_mask_policy_end',
 'county_conditions', 
 'state_mask_policy_start', 
 'state_conditions'
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

def load_county_mask_data(data_path: Path) -> pd.DataFrame:
    # county_mask_policy_end is currently too messy to use, but we should check if it is usable in future
    mask_df = pd.read_csv(data_path, parse_dates=["county_mask_policy_start", "state_mask_policy_start", "state_mask_policy_end"], usecols=county_mask_cols)
    for col in ["_mask_policy_start", "_conditions"]:
        mask_df["county" + col].fillna(mask_df["state" + col], inplace=True)
    mask_df[['state_name']] = mask_df['state_name'].str.title()
    mask_df = mask_df[mask_df['county_fips'] != 'Yes'] # messy data joys!
    mask_df['county_fips'] = pd.to_numeric(mask_df['county_fips'])
    # for now just including "all public places" mask policies as the other variations are complicated 
    return mask_df[mask_df["county_conditions"] == "all public places"]

def load_metro_areas(data_path: Path) -> pd.DataFrame:
    metro_df = pd.read_csv(data_path)
    return metro_df[metro_df["area_type"] == "Metro"][["cbsa_fips", "county_fips", "county_name", "state_codes", "state_name"]]

def fill_dummies(grp, cols):
    for col in cols:
        start_date = grp.index.values[grp[col] == 1]
        grp[col] = grp[col].fillna(0)
        if not start_date.size == 0:
            grp.loc[start_date[0]:, col] = 1
    return grp 

def add_mask_dummies(grp, mask_df):
    start_date = mask_df["county_mask_policy_start"].values[mask_df["county_fips"] == grp["countyfips"].unique()[0]]
    end_date = mask_df["state_mask_policy_end"].values[mask_df["county_fips"] == grp["countyfips"].unique()[0]]
    if not start_date.size == 0:
        if not end_date.size == 0:
            grp.loc[(grp.index.get_level_values(level='date') >= start_date[0]) & (grp.index.get_level_values(level='date') <= end_date[0]), "intervention_mask_all_public"] = 1
        else:
            grp.loc[(grp.index.get_level_values(level='date') >= start_date[0]), "intervention_mask_all_public"] == 1
    return grp
    
def state_level_intervention_data(data_path: Path) -> pd.DataFrame:
    state_policy_df = pd.read_excel(data/"COVID-19 US state policy database_08_03_2020.xlsx", 1)[list(state_interventions_info.keys())]
    state_policy_df = state_policy_df.set_index("STATE").iloc[4:, :].dropna(axis=0)
    return state_policy_df

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



