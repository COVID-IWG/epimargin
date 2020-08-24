from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

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
    "census_fips_code",
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

predictor_cols = ['intervention_>50_gatherings',
 'intervention_>500_gatherings',
 'intervention_Federal_guidelines',
 'intervention_entertainment/gym',
 'intervention_foreign_travel_ban',
 'intervention_public_schools',
 'intervention_restaurant_dine-in',
 'intervention_stay_at_home',
 'intervention_mask_all_public',
 'retail_and_recreation_percent_change_from_baseline',
 'grocery_and_pharmacy_percent_change_from_baseline',
 'parks_percent_change_from_baseline',
 'transit_stations_percent_change_from_baseline',
 'workplaces_percent_change_from_baseline',
 'residential_percent_change_from_baseline']

google_cols = [
    "retail_and_recreation_percent_change_from_baseline", 
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline", 
    "transit_stations_percent_change_from_baseline", 
    "workplaces_percent_change_from_baseline", 
    "residential_percent_change_from_baseline"
]

#Added dummy_cols
dummy_cols = ['intervention_>50 gatherings','intervention_>500 gatherings', 'intervention_Federal guidelines',
    'intervention_entertainment/gym', 'intervention_foreign travel ban', 'intervention_public schools',
    'intervention_restaurant dine-in', 'intervention_stay at home']

##Added drop_county
drop_county = [28163, 47111, 47115, 22037, 51735, 45081, 47129, 21023, 22047, 47153, 47159, 22075, 22077, 47169,
    47173, 22091, 22093, 21077, 21081, 18013, 48229, 22121, 17005, 22125, 21103, 20079, 17013, 51830, 18047, 17027,
    19077, 20107, 19085, 16015, 37007, 40081, 20121, 39073, 37029, 16045, 19121, 40113, 40117, 19129, 17083, 37053,
    18111, 18115, 21191, 16073, 39117, 37073, 39123, 39127, 21215, 20191, 12003, 13033, 13035, 18161, 35057, 36095,
    55049, 13083, 36123, 13085, 31025, 55093, 51007, 53059, 53065, 8014, 8019, 29013, 51036, 13149, 29025, 51043,
    8039, 13159, 51053, 8047, 51570, 13171, 29049, 13181, 28029, 49023, 51075, 49029, 13189, 13199, 48015, 51600,
    28051, 48019, 51097, 51610, 13211, 8093, 51101, 51620, 47015, 51113, 51115, 13227, 51630, 13231, 31155,29107,
    8119, 51127, 5053, 28093, 27079, 47047,31177, 51145, 51157, 27095, 45017, 28127, 51685, 51175, 28137, 45037, 45039,
    1007, 5105, 28143, 51183, 22007, 29177]

def load_country_google_mobility(country_code: str) -> pd.DataFrame:
    full_df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', parse_dates = ["date"])
    country_df = full_df[full_df["country_region_code"] == country_code]
    country_df["sub_region_1"] = country_df["sub_region_1"].str.replace(" and ", " & ")
    return country_df[google_mobility_columns]

def load_us_county_data(file: str, url: Optional[str] = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/") -> pd.DataFrame:
    df = pd.read_csv(url + file, dtype={'countyfips': str})
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["state_name"] = df["state"].map(state_name_lookup)
    df["county_name"] = df["county_name"].str.title()
    return df

def load_intervention_data() -> pd.DataFrame:
    interventions = pd.read_csv('https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/raw_data/national/public_implementations_fips.csv')
    interventions.rename(columns={"Unnamed: 1": "county_name", "Unnamed: 2": "state", "FIPS": "countyfips"}, inplace=True)
    interventions["county_name"] = interventions["county_name"].str.title()
    interventions = pd.DataFrame(interventions.set_index(["state", "countyfips"]).iloc[:, 1:].stack(), columns=["date"]).reset_index().rename(columns={"level_2": "intervention"})    
    interventions["state_name"] = interventions["state"].map(state_name_lookup)
    interventions["date"] = pd.to_datetime(interventions["date"]+ "-2020", format="%d-%b-%Y")
    interventions = interventions.set_index(["state_name", "countyfips", "date"]).iloc[:,1:].sort_index()
    
    #Adding below code
    interventions_dummy = pd.get_dummies(interventions).reset_index()

    #return pd.get_dummies(interventions) - commented line
    return interventions_dummy.groupby(["state_name", "countyfips", "date"])[dummy_cols].sum()

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
    mask_df['countyfips'] = pd.to_numeric(mask_df['county_fips'])
    # for now just including "all public places" mask policies as the other variations are complicated 
    return mask_df[mask_df["county_conditions"] == "all public places"]

def load_metro_areas(data_path: Path) -> pd.DataFrame:
    metro_df = pd.read_csv(data_path)
    metro_df.rename(columns={"county_fips": "countyfips"}, inplace=True)
    return metro_df[metro_df["area_type"] == "Metro"][["cbsa_fips", "countyfips", "county_name", "state_codes", "state_name"]]

def fill_dummies(grp, cols):
    for col in cols:
        start_date = grp.index.values[grp[col] == 1]
        grp[col] = grp[col].fillna(0)
        if not start_date.size == 0:
            grp.loc[start_date[0]:, col] = 1
    return grp 

def add_mask_dummies(grp, mask_df):
    start_date = mask_df["county_mask_policy_start"].values[mask_df["countyfips"] == grp.name[1]]
    end_date = mask_df["state_mask_policy_end"].values[mask_df["countyfips"] == grp.name[1]]
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
    county_cases = pd.DataFrame(case_df.set_index(["state_name", "countyfips"]).iloc[:,3:].stack()).rename(columns={0:"cumulative_confirmed_cases"}).reset_index()
    county_cases["date"] = pd.to_datetime(county_cases["level_2"],format="%m/%d/%y")
    county_cases = county_cases.set_index(["state_name", "countyfips", "date"]).iloc[:,1:]
    county_cases["cumulative_confirmed_cases"] = pd.to_numeric(county_cases["cumulative_confirmed_cases"])
    county_cases = county_cases.groupby(["state_name", "countyfips"]).apply(add_delta_col)
    county_cases["daily_confirmed_cases"].fillna(county_cases["cumulative_confirmed_cases"], inplace=True)
    return county_cases["daily_confirmed_cases"]

def filter_start_outbreak(case_df: pd.DataFrame) -> pd.DataFrame:
    case_df = case_df.reset_index().groupby(['state_name','countyfips']).apply(lambda x: x[x['date'] >= x['date'][x['daily_confirmed_cases'] >= 10].min()])
    return case_df.iloc[:, 2:].reset_index().set_index(['state_name','countyfips','date']).iloc[:, 1:]

def filter_top_metros(case_df: pd.DataFrame, num: Optional[int] = 100) -> pd.DataFrame:
    metro_areas = case_df.reset_index()[['countyfips','cbsa_fips','population']].drop_duplicates()
    metro_pops = pd.DataFrame(metro_areas.groupby('cbsa_fips')['population'].sum()).reset_index()
    metro_pops.sort_values(by='population', ascending=False, inplace=True)
    top_metros = list(metro_pops.iloc[:100,:]['cbsa_fips'])
    return case_df[case_df['cbsa_fips'].isin(top_metros)]

def add_lag_cols(grp: pd.DataFrame, cols: Sequence[str]):
    for lag in [-1, -7, -14]:
        for col in cols:
            grp[col + '_lag_' + str(lag)] = grp[col].shift(lag)
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

def poli_aff(data_path: Path) -> pd.DataFrame:
    vote_df = pd.read_csv(data_path)
    vote16_df = vote_df[vote_df.year==2016]
    
    # Format data
    vote16_df = vote16_df.reset_index(drop=True).fillna(value={"party": "other"})
    vote16_df.drop(columns = ["year", "office", "version"], inplace=True)
    vote16_df["pvote"] = vote16_df.candidatevotes / vote16_df.totalvotes
    # Create pivot table
    tbl_vote16 = pd.pivot_table(vote16_df, values="pvote", index=["state", "state_po", "county", "FIPS"], columns="party", aggfunc=np.sum).reset_index()
    tbl_vote16.rename(columns={"FIPS": "countyfips"}, inplace=True)
    #Create indicator variables
    tbl_vote16["dem_ind"] = tbl_vote16.democrat.apply(lambda x: 0 if x <0.5 else 1)
    tbl_vote16["rep_ind"] = tbl_vote16.republican.apply(lambda x: 0 if x < 0.5 else 1)
    tbl_vote16["oth_ind"] = tbl_vote16.other.apply(lambda x: 0 if x < 0.5 else 1)
    return tbl_vote16

def impute_missing_mobility(county_df: pd.DataFrame) -> pd.DataFrame:
    county_df.reset_index(inplace=True)
    #Drop 133 counties - refer data analysis
    county_remain = county_df[~county_df.countyfips.isin(drop_county)]
    for county in county_remain.countyfips.unique():
        for col in google_cols:
            try:
                county_remain.loc[county_remain.countyfips==county, col] = county_remain.loc[county_remain.countyfips==county, col].interpolate(method="cubic", limit_direction="both", limit_area="inside")
            except:
                continue

    return county_remain

