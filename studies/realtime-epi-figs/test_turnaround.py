import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib

palette = [[0.8423298817793848, 0.8737404427964184, 0.7524954030731037], [0.5815252468131623, 0.7703468311289211, 0.5923205247665932], [0.35935359003014994, 0.6245622005326175, 0.554154071059354], [0.25744332683867743, 0.42368146872794976, 0.5191691971789514], [0.21392162678343224, 0.20848424698401846, 0.3660805512579508]]
color = palette[2]
sns.despine()

cols = ["date_confirmation", "date_admission_hospital"]
tests = pd.read_csv("example_data/clean-outside-hubei.csv", usecols=cols, parse_dates=cols, dayfirst=True).dropna()
tests = tests[~tests.date_admission_hospital.str.contains(" - ")] 
tests.date_admission_hospital = tests.date_admission_hospital.apply(pd.Timestamp)
tests["turnaround"] = (tests.date_confirmation - tests.date_admission_hospital).dt.days
hist = tests.turnaround.value_counts().sort_index()
hist = hist[hist.index >= 0]
dist = hist/hist.sum()

plt.bar(dist.index, dist.values, color = color)
plt.xlabel("delay (days)")
plt.ylabel("percentage of cases")
print(tikzplotlib.get_tikz_code())
plt.show()