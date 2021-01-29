import pandas as pd
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data
from adaptive.utils import setup

data, _ = setup()


paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 18)]
}

for target in paths['v3'] + paths['v4']:
    download_data(data, target)

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)

schema = {"Deceased": "dD", "Recovered": "dR", "Hospitalized": "dT"}
def assemble_time_series(df):
    ts = get_time_series(df)
    deltas = ts[schema.keys()]\
        .rename(columns = schema)
    deltas = deltas.reindex(pd.date_range(deltas.index.min(), deltas.index.max()), fill_value = 0) 
    merged = deltas.merge(
        deltas.cumsum(axis = 0).rename(columns = lambda _: _[1]),
        left_index = True, right_index = True
    ).astype(int)
    merged.index.name   = "date"
    merged.columns.name = None
    return merged

def plus_30_days(date: str) -> pd.Timestamp:
    return pd.Timedelta(30, "d") + pd.Timestamp(date)

# 1: national phase 1 (high-stratum districts)
# https://www.ijmr.org.in/article.asp?issn=0971-5916;year=2020;volume=152;issue=1;spage=48;epage=60;aulast=Murhekar
high_stratum_districts = [
    "Coimbatore", "Chennai", "Buxar", "Ujjain", "Dausa", "Gautam Buddha Nagar", "Patiala", "Krishna", "Jalandhar", "Saharanpur", "Narmada", "Mahisagar", "Dewas",
    "S.P.S. Nellore",   # Sri Potti Sriramulu Nellore
    "Amroha",           # Jyotiba Phule Nagar
    "Kalaburagi",       # Gulbarga
    "Bengaluru Urban",  # Bangalore, based on https://www.icmr.gov.in/pdf/press_realease_files/ICMR_Press_Release_Sero_Surveillance.pdf
]
assemble_time_series(df[df.detected_district.isin(high_stratum_districts)])\
    ["May 11, 2020":plus_30_days("June 4, 2020")]\
    .to_csv(data/"India_1_national_phase_1.csv")

# 2: national phase 2
# high + others
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3715460

zero_stratum_districts = [ "Vizianagaram", "Pakur", "Beed", "Ganjam", "Bijapur", "Balrampur", "Kabeerdham", "Gonda", "Karbi Anglong", "Udalguri", "Kullu", "Latehar", "Chitradurga", "Rayagada", "Alipurduar"]
low_stratum_districts = [ "Alipurduar", "Parbhani", "Nanded", "Madhubani", "Simdega", "Koraput", "Purnia", "Rajsamand", "Bareilly", "Begusarai", "Kurukshetra", "Kamareddy", "Unnao", "Mau", "Kamrup Metropolitan", "Muzaffarpur", "Gurdaspur", "Bankura", "Jhargram",
    "Jangaon",                 # Jangoan
    "Jalore", "Pauri Garhwal", # "Jalore Garhwal" based on https://www.icmr.gov.in/pdf/press_realease_files/ICMR_Press_Release_Sero_Surveillance.pdf
    "Sabarkantha",             # Sabarkantha
]
med_stratum_districts = ["Pulwama", "Tiruvannamalai", "Sangli", "Arwal", "Gwalior", "Auraiya", "Jalgaon", "Ernakulam", "Nalgonda", "Ludhiana", "Surguja", "Palakkad",
    "Ahmednagar",       # Ahmad Nagar
    "Thrissur",         # Thrisur
    "Purba Medinipur",  # Medinipur East
    "South 24 Parganas" # 24 Paraganas South
]

all_stratum_districts = zero_stratum_districts + low_stratum_districts + med_stratum_districts + high_stratum_districts
assemble_time_series(df[df.detected_district.isin(all_stratum_districts)])\
    ["August 17, 2020":plus_30_days("September 22, 2020")]\
    .to_csv(data/"India_2_national_phase_2.csv")

# 3, 4, 5, 6: Delhi
delhi = assemble_time_series(df[df.detected_state == "Delhi"])

for (i, (beg, end)) in enumerate([
    ("June 27, 2020",     "July 10, 2020"),
    ("August 1, 2020",    "August 7, 2020"),
    ("September 1, 2020", "September 7, 2020"),
    ("October 15, 2020",  "October 21, 2020")]):
    delhi[beg:plus_30_days(end)].to_csv(data/f"India_{i+3}_Delhi_phase_{i+1}.csv")

# 7: Karnataka
assemble_time_series(df[df.detected_state == "Karnataka"])\
    ["June 15, 2020":plus_30_days("August 29, 2020")]\
    .to_csv(data/"India_7_Karnataka.csv")

# 8: MH Pimpri-Chinchwad
assemble_time_series(df[df.detected_district == "Pune"])\
    ["October 7, 2020": plus_30_days("October 17, 2020")]\
    .to_csv(data/"India_8_MH_Pimpri_Chinchwad.csv")

# 9, 10: Mumbai phases 1 and 2 
assemble_time_series(df[df.detected_district == "Mumbai"])\
    ["June 29, 2020": plus_30_days("July 19, 2020")]\
    .to_csv(data/"India_9_MH_Mumbai_1.csv")

assemble_time_series(df[df.detected_district == "Mumbai"])\
    ["August 15, 2020": plus_30_days("August 31, 2020")]\
    .to_csv(data/"India_10_MH_Mumbai_2.csv")

# 11: Odisha (Bhubaneswar -> Khordha, Berhampur -> Ganjam, Rourkela -> Sundargarh)
assemble_time_series(df[df.detected_district == "Ganjam"])\
    ["August 6, 2020": plus_30_days("August 6, 2020")]\
    .to_csv(data/"India_11_Odisha_Berhampur.csv")
assemble_time_series(df[df.detected_district == "Khordha"])\
    ["August 26, 2020": plus_30_days("August 26, 2020")]\
    .to_csv(data/"India_11_Odisha_Bhubaneswar.csv")
assemble_time_series(df[df.detected_district == "Sundargarh"])\
    ["September 1, 2020": plus_30_days("September 1, 2020")]\
    .to_csv(data/"India_11_Odisha_Rourkela.csv")

# 12, 13, 14: Puducherry phases 1, 2, 3
assemble_time_series(df[df.detected_state == "Puducherry"])\
    ["August 11, 2020": plus_30_days("August 16, 2020")]\
    .to_csv(data/"India_12_Puducherry_phase_1.csv")
assemble_time_series(df[df.detected_state == "Puducherry"])\
    ["September 10, 2020": plus_30_days("September 16, 2020")]\
    .to_csv(data/"India_13_Puducherry_phase_2.csv")
assemble_time_series(df[df.detected_state == "Puducherry"])\
    ["October 12, 2020": plus_30_days("October 16, 2020")]\
    .to_csv(data/"India_13_Puducherry_phase_3.csv")

#15: Pune
assemble_time_series(df[df.detected_district == "Pune"])\
    ["October 12, 2020": plus_30_days("October 16, 2020")]\
    .to_csv(data/"India_15_Pune.csv")

# 16: WB, Paschim Medinipur
assemble_time_series(df[df.detected_district == "Paschim Medinipur"])\
    ["July 24, 2020": plus_30_days("August 7, 2020")]\
    .to_csv(data/"India_16_West_Bengal_Paschim_Medinipur.csv")

districts = lambda districts: df[df.detected_district.isin(districts)]
districts = lambda districts: df[df.detected_district.isin(districts)]
