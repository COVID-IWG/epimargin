from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import math

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

google_mobility_cols = [
    'retail_and_recreation_percent_change_from_baseline', 
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline', 
    'transit_stations_percent_change_from_baseline', 
    'workplaces_percent_change_from_baseline', 
    'residential_percent_change_from_baseline'
]

# Added columns that have max data between mar-jul
data_cols = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline'
]

state_interventions_info = {
 'STATE': 'State',
 'POSTCODE': 'State Abbreviation',
 'STAYHOME': 'Stay at home/ shelter in place',
 'END_STHM': 'End/relax stay at home/shelter in place',
 'RELIGEX': 'Religious Gatherings Exempt Without Clear Social Distance Mandate*',
 'FM_ALL': 'Mandate face mask use by all individuals in public spaces'
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

def load_country_google_mobility(country_code: str) -> pd.DataFrame:
    cols = ['sub_region_1','census_fips_code','date'] + google_mobility_cols
    full_df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', parse_dates = ['date'])
    country_df = full_df[full_df['country_region_code'] == country_code]
    country_df['sub_region_1'] = country_df['sub_region_1'].str.replace(' and ', ' & ')
    return country_df[cols]

def load_us_county_data(file: str, url: Optional[str] = 'https://usafactsstatic.blob.core.windows.net/public/data/covid-19/') -> pd.DataFrame:
    df = pd.read_csv(url + file, dtype={'countyfips': str})
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['state_name'] = df['state'].map(state_name_lookup)
    df['county_name'] = df['county_name'].str.title()
    return df

def load_metro_areas(data_path: Path) -> pd.DataFrame:
    metro_df = pd.read_csv(data_path)
    metro_df.rename(columns={'county_fips': 'countyfips'}, inplace=True)
    return metro_df[metro_df['area_type'] == 'Metro'][['cbsa_fips', 'countyfips', 'county_name', 'state_codes', 'state_name']]

def fill_dummies(grp, intervention_col: str, start_date_col: str, end_date_col: Optional[str] = None):
    grp[intervention_col] = 0
    start_date = grp[start_date_col].unique()
    end_date = grp[end_date_col].unique() if end_date_col else [None]
    if start_date[0]:
        if end_date[0]:
            grp.loc[(grp.index.get_level_values(level = 'date') >= start_date[0]) & (grp.index.get_level_values(level='date') <= end_date[0]),intervention_col] = 1
        grp.loc[(grp.index.get_level_values(level = 'date') >= start_date[0]),intervention_col] = 1
    return grp 
    
def state_level_intervention_data(data_path: Path) -> pd.DataFrame:
    state_policy_df = pd.read_excel(data_path, 1)[list(state_interventions_info.keys())]
    state_policy_df = state_policy_df.set_index('STATE').iloc[4:, :].dropna(axis=0)
    state_policy_df.columns = ['state','start_stay_at_home','end_stay_at_home','religious_exception_stay_home','mask_mandate_all']
    return state_policy_df

def get_case_timeseries(case_df: pd.DataFrame) -> pd.DataFrame:
    county_cases = pd.DataFrame(case_df.set_index(['state_name','countyfips']).iloc[:,3:].stack()).rename(columns={0:'cumulative_confirmed_cases'}).reset_index()
    county_cases['date'] = pd.to_datetime(county_cases['level_2'],format='%m/%d/%y')
    county_cases = county_cases.set_index(['state_name', 'countyfips', 'date']).iloc[:,1:]
    county_cases['cumulative_confirmed_cases'] = pd.to_numeric(county_cases['cumulative_confirmed_cases'])
    county_cases = county_cases.groupby(['state_name','countyfips']).apply(add_delta_col)
    county_cases['daily_confirmed_cases'].fillna(county_cases['cumulative_confirmed_cases'], inplace=True)
    return pd.DataFrame(county_cases['daily_confirmed_cases'][county_cases.index.get_level_values(level=1) != 0]).reset_index()

def start_outbreak_dummy(case_df: pd.DataFrame) -> pd.DataFrame:
    start_dates = case_df.reset_index().groupby(['metro-state']).apply(lambda x: x['date'][x['daily_confirmed_cases'] >= 10].min())
    case_df = case_df.reset_index().merge(start_dates.reset_index(), on='metro-state', how='outer').rename(columns={0:'metro_outbreak_start'}).set_index(['metro-state', 'date'])
    return case_df.groupby(['metro-state']).apply(fill_outbreak_dummy)

def fill_outbreak_dummy(grp):
    start_date = grp['metro_outbreak_start'].unique()
    grp['threshold_ind'] = 0
    if not start_date.size == 0:
        # 14 days before start of outbreak to try to capture data before interventions were put in place
        threshold_start = start_date[0] - pd.Timedelta(14, unit='d')
        grp.loc[(grp.index.get_level_values(level='date') >= threshold_start), 'threshold_ind'] = 1
    return grp

def get_top_metros(county_populations: pd.DataFrame, metros, n: Optional[int] = 100) -> pd.DataFrame:
    metro_areas = metros.merge(county_populations[['countyfips','population']], on='countyfips')
    metros = pd.DataFrame(metro_areas.groupby('cbsa_fips')['population'].sum()).reset_index().merge(metro_areas[['cbsa_fips','state']], on='cbsa_fips')
    top_n_metros = list(metro_areas.sort_values(by='population', ascending=False).iloc[:n,:]['cbsa_fips'])
    return metro_areas[['countyfips','state','cbsa_fips']][metro_areas['cbsa_fips'].isin(top_n_metros)]

def get_metro_dummies(county_df):
    metro_dummies = pd.get_dummies(county_df['cbsa_fips'])
    metro_dummies.columns = ['metro_' + str(x) for x in metro_dummies.columns]
    return county_df.join(metro_dummies)

def add_lag_cols(grp: pd.DataFrame, cols: Sequence[str]):
    for lag in [-1, -7, -14]:
        for col in cols:
            grp[col + '_lag_' + str(lag)] = grp[col].shift(lag)
    return grp

def metro_state_mobility_agg(county_mobility: pd.DataFrame) -> pd.DataFrame:
    return county_mobility.groupby(['metro-state', 'date']).apply(weighted, google_mobility_cols)

def weighted(grp, cols):
    return pd.Series(np.average(grp[cols], weights=grp['pop_prop'], axis=0), cols)

def pop_prop_col(grp):
    grp['pop_prop'] = grp['population'] / grp['population'].sum()
    return grp

def add_delta_col(grp: pd.DataFrame):
    grp['daily_confirmed_cases'] = grp['cumulative_confirmed_cases'].diff()
    return grp

def poli_aff(data_path: Path) -> pd.DataFrame:
    vote_df = pd.read_csv(data_path)
    vote16_df = vote_df[vote_df.year==2016]
    # Format data
    vote16_df = vote16_df.reset_index(drop=True).fillna(value={'party': 'other'})
    vote16_df.drop(columns = ['year', 'office', 'version'], inplace=True)
    vote16_df['pvote'] = vote16_df.candidatevotes / vote16_df.totalvotes
    # Create pivot table
    tbl_vote16 = pd.pivot_table(vote16_df, values='pvote', index=['state', 'state_po', 'county', 'FIPS'], columns='party', aggfunc=np.sum).reset_index()
    tbl_vote16.rename(columns={'FIPS': 'countyfips'}, inplace=True)
    #Create indicator variables
    tbl_vote16['dem_ind'] = tbl_vote16.democrat.apply(lambda x: 0 if x <0.5 else 1)
    tbl_vote16['rep_ind'] = tbl_vote16.republican.apply(lambda x: 0 if x < 0.5 else 1)
    tbl_vote16['oth_ind'] = tbl_vote16.other.apply(lambda x: 0 if x < 0.5 else 1)
    return tbl_vote16

def impute_missing_mobility(county_df: pd.DataFrame) -> pd.DataFrame:
    county_df.reset_index(inplace=True)
    for county in county_df.countyfips.unique():
        for col in google_mobility_cols:
            try:
                county_df.loc[county_df.countyfips==county, col] = county_df.loc[county_df.countyfips==county, col].interpolate(method='cubic', limit_direction='both', limit_area='inside')
            except:
                continue
    return county_df

def remain_county(county_df: pd.DataFrame, pct_data: Optional[float] = 0.4, mth: Optional[list] = [3,4,5,6,7]) -> pd.DataFrame:
    county_df['month'] = county_df.date.apply(lambda x: x.month)
    month_df = county_df[county_df.month.isin(mth)].reset_index(drop=True)
    cnt_df = month_df.groupby(['countyfips', 'population'])[data_cols].count().reset_index()
    county_ls_total = []    
    for col in data_cols:
        county_ls = list(cnt_df[cnt_df[col] <= math.ceil(month_df.date.nunique() * pct_data)]['countyfips'])
        county_ls_total.extend(county_ls)
    drop_cnt = list(set(county_ls_total)) 
    return county_df[~county_df.countyfips.isin(drop_cnt)].reset_index(drop=True), drop_cnt
