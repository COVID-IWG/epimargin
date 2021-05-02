import pandas as pd 

districts = {
    ('Tamil Nadu', 'Chennai'): 'Chennai',
    ('West Bengal', 'Paschim Medinipur'): 'Paschim Medinipur',
    ('Maharashtra', 'Pune'): 'Pune',
    ('Maharashtra', 'Mumbai'): 'Mumbai',
    ('Odisha', 'Khordha'): 'Bhubaneswar',
    ('Odisha', 'Ganjam'): 'Berhampur',
    ('Odisha', 'Sundargarh'): 'Rourkela'
}

agebin_labels = ["0-17", "18-29","30-39", "40-49", "50-59", "60-69","70+"]


district_age_pop = pd.read_csv("data/all_india_sero_pop.csv")\
    .set_index(["state", "district"])\
    .filter(like = "N_", axis = 1)

district_age_pop\
    .filter(items = districts.keys(), axis = 0)\
    .astype(int)\
    .assign(sero_location = districts.values())\
    .rename(columns = {f"N_{i}": agebin_labels[i] for i in range(7)})\
    [["sero_location"] + agebin_labels + ["N_tot"]]\
    .to_csv("data/missing_india_sero_pop.csv")
