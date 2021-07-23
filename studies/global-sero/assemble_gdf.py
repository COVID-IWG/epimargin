from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

data = Path.home() / "Downloads"

shp_dst  = lambda c: data / f"gadm36_{c}_shp"
load     = lambda c: gpd.read_file(max(shp_dst(c).glob("*shp")))

ARG = load("ARG")
HUN = load("HUN")
LAO = load("LAO")
PER = load("PER")
BRA = load("BRA")
ZAF = load("ZAF")
CHN = load("CHN")
IND = load("IND")
MOZ = load("MOZ")
IRN = load("IRN")
NPL = load("NPL")
ETH = load("ETH")
KEN = load("KEN")
PAK = load("PAK")
MEX = gpd.read_file(data / "gadm36_MEX_gpkg" / "gadm36_MEX.gpkg")
COL = gpd.read_file(data / "COL_adm_shp"     / "COL_adm2.shp")

queries = [
#    id   country         location                            shp  col       filter                    display_as_point
    (38,  "Argentina",    "Buenos Aires City",                ARG, "NAME_1", "Ciudad de Buenos Aires", True),
    (40,  "Argentina",    "Municipality of Hurlingham",       ARG, "GID_2" , "ARG.1.82_1",             True),
    (5,   "Brazil",       "Maranhao",                         BRA, "GID_1",  "BRA.10_1",               False),
    (73,  "Brazil",       "Sao Paulo City",                   BRA, "NAME_3", "Se",                     True),
    (86,  "Brazil",       "Sergipe",                          BRA, "NAME_1", "Sergipe",                False),
    (75,  "China",        "Wuhan study #2",                   CHN, "NAME_2", "Wuhan",                  True),
    (121, "China",        "Wuhan #1",                         CHN, "NAME_2", "Wuhan",                  True),
    (123, "China",        "Hubei province (excluding Wuhan)", CHN, "NAME_1", "Hubei",                  False),
    (44,  "Colombia",     "Leticia",                          COL, "NAME_2", "Leticia",                True),
    (45,  "Colombia",     "Barranquilla",                     COL, "NAME_2", "Barranquilla",           True),
    (46,  "Colombia",     "Medellín",                         COL, "ID_2",   76,                       True),
    (47,  "Colombia",     "Monteria",                         COL, "ID_2",   333,                      True),
    (48,  "Colombia",     "Bucaramanga",                      COL, "NAME_2", "Bucaramanga",            True),
    (49,  "Colombia",     "Cucuta",                           COL, "ID_2",   805,                      True),
    (50,  "Colombia",     "Villavicencio",                    COL, "NAME_2", "Villavicencio",          True),
    (51,  "Colombia",     "Bogotá",                           COL, "ID_2",   568,                      True),
    (52,  "Colombia",     "Cali",                             COL, "ID_2",   1043,                     True),
    (53,  "Colombia",     "Guapi",                            COL, "ID_2",   418,                      True),
    (54,  "Colombia",     "Ipiales",                          COL, "NAME_2", "Ipiales",                True),
    (21,  "Ethiopia",     "Addis Ababa #1",                   ETH, "NAME_2", "Addis Abeba",            True),
    (80,  "Ethiopia",     "Addis Ababa #2",                   ETH, "NAME_2", "Addis Abeba",            True),
    (192, "Hungary",      "National Study",                   HUN, "NAME_0", "Hungary",                False),
    (10,  "India",        "National",                         IND, "NAME_0", "India",                  False),
    (11,  "India",        "Delhi",                            IND ,"NAME_1", "NCT of Delhi",           True),
    (13,  "India",        "Pimpri-Chinchwad",                 IND, "GID_2",  "IND.20.26_1",            True),
    (16,  "India",        "Puducherry",                       IND, "NAME_1", "Puducherry",             False),
    (19,  "India",        "Chennai",                          IND, "NAME_2", "Chennai",                True),
    (20,  "India",        "Tamil Nadu",                       IND, "NAME_1", "Tamil Nadu",             False),
    (76,  "India",        "Kashmir (whole region)",           IND, "GID_1",  "IND.14_1",               False),
    (77,  "India",        "Kashmir: Srinagar district",       IND, "GID_2",  "IND.14.21_1",            False),
    (84,  "India",        "Odisha: Berhampur",                IND, "NAME_2", "Ganjam",                 True),
    (81,  "Iran",         "National Study",                   IRN, "NAME_0", "Iran",                   False),
    (36,  "Kenya",        "Nairobi County",                   KEN, "NAME_1", "Nairobi",                True),
    (92,  "Laos",         "National Study",                   LAO, "NAME_0", "Laos",                   False),
    (82,  "Mexico",       "National Study",                   MEX, "NAME_0", "Mexico",                 False),
    (23,  "Mozambique",   "Lichinga",                         MOZ, "NAME_3", "Cidade De Lichinga",     True),
    (78,  "Nepal",        "National Study",                   NPL, "NAME_0", "Nepal",                  False),
    (125, "Pakistan",     "Karachi",                          PAK, "NAME_2", "Karachi",                True),
    (126, "Pakistan",     "Lahore",                           PAK, "NAME_3", "Lahore",                 True),
    (59,  "Peru",         "Cusco Province",                   PER, "NAME_1", "Cusco",                  False),
    (63,  "Peru",         "Lima (Metropolitana) + Callao",    PER, "NAME_3", "Lima",                   True),
    (64,  "Peru",         "Iquitos, Loreto",                  PER, "NAME_3", "Iquitos",                True),
    (90,  "South Africa", "Mitchells Plain",                  ZAF, "NAME_3", "City of Cape Town",      True)
]

geometries = [
    df[df[col] == val]\
        .dissolve(by = col)\
        [["geometry"]]\
        .assign(location_id = loc_id, country = country, location = location, display_as_pt = as_pt)\
        .reset_index()\
        .drop(columns = [col])
    for (loc_id, country, location, df, col, val, as_pt) in tqdm(queries)
]
gdf = gpd.GeoDataFrame(pd.concat(geometries, axis = 0))
gdf.to_csv("./data/IFR_geometries.csv")
