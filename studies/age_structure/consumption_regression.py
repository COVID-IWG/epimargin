datareg_sep2020 = pd.read_stata(data/"datareg_sep2020.dta")\
    .dropna()\
    .drop(columns = ["_merge"])