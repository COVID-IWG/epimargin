from adaptive.etl.csse import *
from adaptive.utils import setup
data, _ = setup()

csse_dir = data/"csse"
csse_dir.mkdir(exist_ok = True)

brazil = data/"brazil"
brazil.mkdir(exist_ok = True)

fetch_range(csse_dir, "May 12, 2020", "December 31, 2020")

df = load(csse_dir, "May 12, 2020", "December 31, 2020", "Country_Region == 'Brazil'")\
    .drop(columns = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incidence_Rate", "Case-Fatality_Ratio", "Country_Region"])
    # .drop(columns = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incident_Rate", "Case_Fatality_Ratio", "Country_Region"])
df["Active"] = df["Active"].astype(int)

# national-level 
natl = df.set_index(["date", "Province_State"])\
    .stack()\
    .sum(level = [0, 2])\
    .unstack()\
    [["Deaths", "Recovered", "Confirmed"]]\
    .rename(columns = {"Confirmed": "T", "Deaths": "D", "Recovered": "R"})
diffs = natl.diff().dropna().astype(int).rename(lambda x: "d" + x, axis = 1)
joined = pd.concat([diffs, natl], axis = 1).dropna().astype(int)

joined["May 14, 2020": "June 21, 2020"]\
    .to_csv(brazil/"Brazil_1_national_phase_1.csv")

joined["June 4, 2020": "July 7, 2020"]\
    .to_csv(brazil/"Brazil_2_national_phase_2.csv")

joined["June 21, 2020": "July 24, 2020"]\
    .to_csv(brazil/"Brazil_3_national_phase_3.csv")

def assemble(province: str):
    totals = df[df.Province_State == province]\
        [["date", "Deaths", "Recovered", "Confirmed"]]\
        .set_index("date")\
        .rename(columns = {"Confirmed": "T", "Deaths": "D", "Recovered": "R"})
    diffs = totals.diff().dropna().astype(int).rename(lambda x: "d" + x, axis = 1)
    return pd.concat([totals, diffs], axis = 1).dropna().astype(int)    

# espirito santo 
ES = assemble("Espirito Santo")
for (i, (beg, end)) in enumerate([
    ("May 13, 2020", "June 15, 2020"),
    ("May 27, 2020", "June 29, 2020"),
    ("June 8, 2020", "July 10, 2020"),
    ("June 22, 2020", "July 24, 2020"),
    ("July 27, 2020", "July 29, 2020")
]):
    ES[beg:end].to_csv(brazil/f"Brazil_{i+6}_Espirito_Santo_phase_{i+1}.csv")

# maranhao
MR = assemble("Maranhao")
for (i, (beg, end)) in enumerate([
    ("July 27, 2020", "September 8, 2020"),
    ("October 19, 2020", "November 30, 2020"),
]):
    MR[beg:end].to_csv(brazil/f"Brazil_{i+13}_Maranhao_phase_{i+1}.csv")

# piaui
PI = assemble("Piaui")
for (i, (beg, end)) in enumerate([
    ("April 25, 2020", "May 28, 2020"),
    ("May 6, 2020", "June 9, 2020"),
    ("May 12, 2020", "June 15, 2020"),
    ("May 20, 2020", "June 23, 2020"),
    ("May 30, 2020", "July 2, 2020"),
    ("June 10, 2020", "July 13, 2020"),
    ("June 17, 2020", "July 20, 2020"),
    ("June 27, 2020", "July 30, 2020"),
    ("July 15, 2020", "August 18, 2020"),
]):
    PI[beg:end].to_csv(brazil/f"Brazil_{i+15}_Piaui_phase_{i+1}.csv")
