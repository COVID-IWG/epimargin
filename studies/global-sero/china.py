from adaptive.etl.csse import *
from adaptive.utils import setup
data, _ = setup()

csse_dir = data/"csse"
csse_dir.mkdir(exist_ok = True)

china = data/"china"
china.mkdir(exist_ok = True)

fetch_range(csse_dir, "April 13, 2020", "June 30, 2020")

df = load(csse_dir, "April 13, 2020", "June 30, 2020", "Country_Region == 'China'")\
    .drop(columns = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incidence_Rate", "Case-Fatality_Ratio", "Country_Region"])\
    .assign(Active = lambda _:_["Active"].astype(int))
    # .drop(columns = ["FIPS", "Admin2", "Last_Update", "Lat", "Long_", "Combined_Key", "Incident_Rate", "Case_Fatality_Ratio", "Country_Region"])
# df["Active"] = df["Active"].astype(int)

def assemble(province: str):
    totals = df[df.Province_State == province]\
        [["date", "Deaths", "Recovered", "Confirmed"]]\
        .set_index("date")\
        .rename(columns = {"Confirmed": "T", "Deaths": "D", "Recovered": "R"})
    diffs = totals.diff().dropna().astype(int).rename(lambda x: "d" + x, axis = 1)
    return pd.concat([totals, diffs], axis = 1).dropna().astype(int)    

assemble("Beijing")["April 15, 2020":"May 18, 2020"].to_csv(china/"China_1_Beijing.csv")