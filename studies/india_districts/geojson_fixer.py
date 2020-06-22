from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

state_codes = {
    "Andaman and Nicobar Islands": "AN",
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BR",
    "Chandigarh": "CH",
    "Chhattisgarh": "CT",
    "Dadra and Nagar Haveli and Daman and Diu": "DNDD",
    "Delhi": "DL",
    "Goa": "GA",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jammu and Kashmir": "JK",
    "Jharkhand": "JH",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Lakshadweep": "LD",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "ML",
    "Mizoram": "MZ",
    "Nagaland": "NL",
    "Odisha": "OR",
    "Puducherry": "PY",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TG",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttarakhand": "UT",
    "West Bengal": "WB",
}


root = Path(__file__).parent 
data = root/"data"

for gdf_path in data.glob("gj_raw/*.json"):
    gdf = gpd.read_file(gdf_path)
    state = gdf.st_nm[0]
    district_filename = data/f"website_{state}_districts.csv"
    if district_filename.exists() and state in state_codes.keys():
        districts = pd.read_csv(district_filename)
        merged = gdf.merge(
            districts.drop(columns="Unnamed: 0").fillna(""), 
            on="district", how="left"
        ).drop(columns=["dt_code", "st_code", "year"])
        code = state_codes[state]
        filename = data/f"gj_fix/{code}.json"
        merged.to_file(filename, driver="GeoJSON")

india = gpd.read_file(data/"gj_raw/india.json")\
           .dissolve(by="st_nm")\
           .drop(columns = ["id", "district", "dt_code", "st_code", "year"])\
           .reset_index()\
           .rename(columns = {"st_nm" : "district"})

Rstate = []
for state in india.district: 
    print(state)
    try:
        est = pd.read_csv(data/f"website_{state}_est.csv")
        R = est.iloc[-1].Rt
    except FileNotFoundError:
        print("  file not found for est")
        R = np.nan
    try:
        proj = pd.read_csv(data/f"website_{state}_rtproj.csv")
        Rp = proj.iloc[-1].Rt_proj
    except FileNotFoundError:
        print("  file not found for proj")
        Rp = np.nan

    Rstate.append((state, R, Rp))

Rstate = pd.DataFrame(Rstate, columns = ["district", "Rt", "Rt_proj"])

india_merge = india.merge(Rstate, on="district")
india_merge.to_file(data/"gj_fix/IN.json", driver="GeoJSON")