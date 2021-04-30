from pathlib import Path

import epimargin.plots as plt
import numpy as np
import pandas as pd
from epimargin.smoothing import convolution
from matplotlib.dates import DateFormatter

root    = Path.home() / "Dropbox" / "Covid"
results = root / "results"
data    = Path("./data")

plt.set_theme("substack")

formatter = DateFormatter("%b\n%Y")

# fig 1 
meta_ifrs               = pd.read_stata(data / "meta_ifrs.dta")
all_location_comparison = pd.read_stata(data / "all_location_comparison.dta")

## male 
levin_male = meta_ifrs[(meta_ifrs.location == "levin") & (meta_ifrs.male == 1.0)].query('age <= 70').query('age >= 10')
od_male    = meta_ifrs[(meta_ifrs.location == "od")    & (meta_ifrs.male == 1.0)].query('age <= 70').query('age >= 10')
cai_bihar_male     = all_location_comparison[(all_location_comparison.location == "Bihar")      & (all_location_comparison.male == 1.0)].query('ifr > 0')
cai_mumbai_male    = all_location_comparison[(all_location_comparison.location == "Mumbai")     & (all_location_comparison.male == 1.0)].sort_values("age_bin_pooled")
cai_karnataka_male = all_location_comparison[(all_location_comparison.location == "Karnataka")  & (all_location_comparison.male == 1.0)]
cai_tamilnadu_male = all_location_comparison[(all_location_comparison.location == "tamil_nadu") & (all_location_comparison.male == 1.0)]

plt.plot(levin_male.age, (levin_male.ifr), color = "gray", linestyle = "dotted", label = "meta (Levin et al. 2020; slope = 0.123)")
plt.plot(od_male.age,    (od_male.ifr),   color = "black", label = "meta (O'Driscoll et al. 2020; slope = 0.114) ")
plt.plot(cai_bihar_male.age_bin_pooled,     (100 * cai_bihar_male.ifr),     color = "#007CD4", label = "Bihar (Cai et al. 2021; slope = 0.047)")
plt.plot(cai_mumbai_male.age_bin_pooled,    (100 * cai_mumbai_male.ifr),    color = "#FF0000", label = "Mumbai (Cai et al. 2021; slope = 0.100)")
plt.plot(cai_karnataka_male.age_bin_pooled, (100 * cai_karnataka_male.ifr), color = "#55A423", label = "Karnataka (Cai et al. 2021; slope = 0.103)")
plt.plot(cai_tamilnadu_male.age_bin_pooled, (100 * cai_tamilnadu_male.ifr), color = "#FFB700", label = "Tamil Nadu (Cai et al. 2021; slope = 0.089)")
plt.legend(prop = plt.theme.note, handlelength = 1, framealpha = 0)
plt.PlotDevice().axis_labels(x = "age", y = "IFR (%; log-scale)").l_title("IFRs by age (male)").adjust(bottom = 0.16)
plt.semilogy()
plt.show()

##
levin_female = meta_ifrs[(meta_ifrs.location == "levin") & (meta_ifrs.male == 0.0)].query('age <= 70').query('age >= 20')
od_female    = meta_ifrs[(meta_ifrs.location == "od")    & (meta_ifrs.male == 0.0)].query('age <= 70').query('age >= 20')
cai_bihar_female     = all_location_comparison[(all_location_comparison.location == "Bihar")      & (all_location_comparison.male == 0.0)]
cai_mumbai_female    = all_location_comparison[(all_location_comparison.location == "Mumbai")     & (all_location_comparison.male == 0.0)].sort_values("age_bin_pooled")
cai_karnataka_female = all_location_comparison[(all_location_comparison.location == "Karnataka")  & (all_location_comparison.male == 0.0)]
cai_tamilnadu_female = all_location_comparison[(all_location_comparison.location == "tamil_nadu") & (all_location_comparison.male == 0.0)]

plt.plot(levin_female.age, (levin_female.ifr), color = "gray", linestyle = "dotted", label = "meta (Levin et al. 2020; slope = 0.123)")
plt.plot(od_female.age,    (od_female.ifr),   color = "black", label = "meta (O'Driscoll et al. 2020; slope = 0.111) ")
plt.plot(cai_mumbai_female.age_bin_pooled,    (100 * cai_mumbai_female.ifr),    color = "#FF0000", label = "Mumbai (Cai et al. 2021; slope = 0.106)")
plt.plot(cai_karnataka_female.age_bin_pooled, (100 * cai_karnataka_female.ifr), color = "#55A423", label = "Karnataka (Cai et al. 2021; slope = 0.087)")
plt.plot(cai_tamilnadu_female.age_bin_pooled, (100 * cai_tamilnadu_female.ifr), color = "#FFB700", label = "Tamil Nadu (Cai et al. 2021; slope = 0.072)")
plt.legend(prop = plt.theme.note, handlelength = 1, framealpha = 0)
plt.PlotDevice().axis_labels(x = "age", y = "IFR (%; log-scale)").l_title("IFRs by age (female)").adjust(bottom = 0.16)
plt.semilogy()
plt.show()



# fig 2
age_IFR = pd.read_excel(results / "age_IFR.xlsx")\
    .dropna()\
    .set_index("Age group")

## linear 
plt.plot(age_IFR["CFR 2020 (adjusted for reporting)"], color = "blue",   label = "2020")
plt.plot(age_IFR["CFR 2021 (adjusted for reporting)"], color = "orange", label = "2021")
plt.legend(prop = plt.theme.label, handlelength = 1, framealpha = 0)
plt.PlotDevice()\
    .axis_labels(x = "age group", y = "CFR")\
    .l_title("CFR in India (adjusted for reporting)")\
    .r_title("source:\nICMR")\
    .adjust(left = 0.11, bottom = 0.15, right = 0.95)
plt.show()

## log 
plt.plot(age_IFR["CFR 2020 (adjusted for reporting)"], color = "blue",   label = "2020")
plt.plot(age_IFR["CFR 2021 (adjusted for reporting)"], color = "orange", label = "2021")
plt.legend(prop = plt.theme.label, handlelength = 1, framealpha = 0)
plt.PlotDevice()\
    .axis_labels(x = "age group", y = "CFR (log-scaled)")\
    .l_title("CFR in India (adjusted for reporting)")\
    .r_title("source:\nICMR")\
    .adjust(left = 0.11, bottom = 0.15, right = 0.95)
plt.semilogy()
plt.show()


# fig 3
india_data = pd.read_csv(results / "india_data.csv", parse_dates = ["dt"])\
    .query("State == 'TT'")\
    .set_index("dt")\
    .sort_index()

fig, axs = plt.subplots(2, 2, sharex = True, sharey = True)

plt.sca(axs[0, 0])
plt.scatter(india_data.index, india_data["cfr_2week"], color = "black", s = 2)
plt.title("2-week lag", loc = "left", fontdict = plt.theme.label)

plt.sca(axs[0, 1])
plt.scatter(india_data.index, india_data["cfr_maxcor"], color = "black", s = 2)
plt.title("10-day lag", loc = "left", fontdict = plt.theme.label)

plt.sca(axs[1, 0])
plt.scatter(india_data.index, india_data["cfr_1week"], color = "black", s = 2)
plt.title("1-week lag", loc = "left", fontdict = plt.theme.label)

plt.sca(axs[1, 1])
plt.scatter(india_data.index, india_data["cfr_same"], color = "black", s = 2)
plt.title("no lag", loc = "left", fontdict = plt.theme.label)
plt.gca().xaxis.set_major_formatter(formatter)

plt.PlotDevice()\
    .title(f"CFR: varying lag between cases & deaths\n", ha = "center", x = 0.5)\
    .adjust(left = 0.10, bottom = 0.125, right = 0.96, top = 0.85, hspace = 0.27)
fig.text(0.04, 0.45, "CFR", rotation = "vertical", ha = "center", va = "bottom", fontdict = plt.theme.label, color = "dimgray")
fig.text(0.525, 0.01, "\ndate", ha = "center", va = "bottom", fontdict = plt.theme.label, color = "dimgray")
plt.show()


# fig 4

## a
plt.plot(india_data.tests_7day["July 01, 2020":]/(1e6), color = "darkred")
plt.PlotDevice()\
    .l_title("tests per day")\
    .axis_labels(x = "date", y = "tests (millions)")\
    .adjust(bottom = 0.16)
plt.gca().xaxis.set_major_formatter(formatter)
plt.show()

## b
plt.plot(india_data.tpr["July 01, 2020":], color = "royalblue")
plt.PlotDevice()\
    .l_title("test positivity rate over time")\
    .axis_labels(x = "date", y = "TPR (%)")\
    .adjust(bottom = 0.16)
plt.gca().xaxis.set_major_formatter(formatter)
plt.show()

# fig 5
smoothing = convolution("uniform", window = 7)
plt.plot(india_data.index, smoothing(india_data["cfr_maxcor"]), color = "maroon", label = "10-day lag CFR (upper bound)" )
plt.plot(india_data.index, 
    (smoothing(india_data["cfr_maxcor"]) + smoothing(india_data["adj_cfr"]))/2, 
    color = "indigo", linestyle = "dashed", label = "averaged")
plt.plot(india_data.index, smoothing(india_data["adj_cfr"]),    color = "navy",   label = "adjusted CFR (lower bound)" )
plt.fill_between(
    india_data.index, 
    smoothing(india_data["adj_cfr"]),
    smoothing(india_data["cfr_maxcor"]),
    color = "indigo", alpha = 0.1
)
plt.legend(prop = plt.theme.label, handlelength = 1, framealpha = 0)
plt.gca().xaxis.set_major_formatter(formatter)
plt.PlotDevice().l_title("range of CFR over time").axis_labels("date", "CFR").adjust(bottom = 0.16)
plt.show()
