import pandas as pd

dropbox_url = "https://www.dropbox.com/s/exolv53z8n53kq7/TNSS_CleanDeathData_updated.csv?dl=1"
death_cutoff = pd.Timestamp("December 7, 2020")
age_cutoffs = [0, 18, 30, 40, 50, 60, 70, 103] # topcode deaths > 100 as = 100

# download and read data 
tn = pd.read_csv(dropbox_url, parse_dates = ["dateofdeath"])\
    [["age", "dateofdeath"]]
tn = tn[tn.dateofdeath <= death_cutoff]

# resolve non-numeric labels:
tn["age"][~tn["age"].str.isdigit()] = 1
tn["age"] = tn["age"].astype(float)

tn["agebin"] = pd.cut(tn["age"], age_cutoffs, right = False)

hist = tn.groupby("agebin").size()
print(hist)
print(hist.sum())
