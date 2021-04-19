from glob import glob
from itertools import groupby
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib

# estimates testing delay for NYC, Aug 2020

# instructions: 
# 1. clone https://github.com/nychealth/coronavirus-data
# 2. collect the diffs for august:
# echo """ 8.31 6fa5de8 bd0a5fe
# 8.30 35ab5bf 6fa5de8
# 8.29 d85359c 35ab5bf
# 8.28 2235426 d85359c
# 8.27 873753a 2235426
# 8.26 13daf0b 873753a
# 8.25 855b4e1 13daf0b
# 8.24 96f12ea 855b4e1
# 8.23 7e3f387 96f12ea
# 8.22 ce3f3a6 7e3f387
# 8.21 647a821 ce3f3a6
# 8.20 5d65451 647a821
# 8.19 02b2578 5d65451
# 8.18 0f4f005 02b2578
# 8.17 a5fecbe 0f4f005
# 8.16 91bd862 a5fecbe
# 8.15 538891f 91bd862
# 8.14 5f06bd0 538891f
# 8.13 16224c0 5f06bd0
# 8.12 5158a16 16224c0
# 8.11 f4f4335 5158a16
# 8.10 66105d1 f4f4335
# 8.09 7c5b81b 66105d1
# 8.08 dcdc459 7c5b81b
# 8.07 cf02bdf dcdc459
# 8.06 b6a811e cf02bdf
# 8.05 2f12509 b6a811e
# 8.04 773ffb0 2f12509
# 8.03 a7315f6 773ffb0
# 8.02 6a26616 a7315f6
# 8.01 f10021a 6a26616
# """ | while read day hash1 hash2; do
#         git diff --no-prefix ${hash1} ${hash2} case-hosp-death.csv > ${day}.diff
# done

# 3. analyze the *.diff files

palette = [[0.8423298817793848, 0.8737404427964184, 0.7524954030731037], [0.5815252468131623, 0.7703468311289211, 0.5923205247665932], [0.35935359003014994, 0.6245622005326175, 0.554154071059354], [0.25744332683867743, 0.42368146872794976, 0.5191691971789514], [0.21392162678343224, 0.20848424698401846, 0.3660805512579508]]
color = palette[-2]

def load_diff(histogram: Dict[int, float], filename: str):
    timestamp = filename.rsplit("d", 1)[0] + "2020"
    with open(filename) as fp:
        # filter to lines that are part of the diff, and only look at hospitalizations
        diff = list(map(lambda l: l.split(",")[:2], filter(lambda l: l[0] in "+-", fp.readlines()))) 
        # group by date and only look at aug updates
        for (key, group) in groupby(sorted(diff, key = lambda t: t[0][1:]), lambda t: t[0][1:]):
            updates = list(group)
            if key.startswith("08"):
                if len(updates) > 1:
                    cases = int(updates[1][1]) - int(updates[0][1])
                else: 
                    cases = int(updates[0][1])
                delay = (pd.Timestamp(timestamp) - pd.Timestamp(key)).days - 1 # day N testing data reflected in day N+1 repository update
                histogram[delay] = cases + histogram.get(delay, 0)

histogram = dict()
for filename in glob("*.diff"):
    load_diff(histogram, filename)

clipped_histogram = {k: v for (k, v) in {k: max(0, v) for (k, v) in histogram.items()}.items() if v > 0}
norm = sum(clipped_histogram.values())
distribution = {k: v/norm for (k, v) in clipped_histogram.items()}

d = sorted(distribution.keys())
h = [100*distribution[_] for _ in d]
plt.bar(d, h, color = color)
plt.ylabel("percentage of cases")
plt.xlabel("delay (days)")
print(tikzplotlib.get_tikz_code())
plt.show()

#mean 
sum(k*v for (k, v) in distribution.items())
