from itertools import product
from studies.age_structure.national_model import D

import pandas as pd
from scipy.stats.morestats import probplot
from studies.age_structure.commons import *

""" Calculate willingness to pay """ 

coeffs = pd.read_stata(data/"reg_estimates.dta")\
    [["parm", "label", "estimate"]]\
    .set_index("parm")\
    .rename(columns = {"parm": "param", "estimate": "value"})

pcons = pd.read_stata("data/pcons_2019m6.dta").set_index("districtnum")

datareg = pd.read_stata("data/datareg.dta")
IN_simulated_percap = pd.read_csv("data/IN_simulated_percap.csv") 

I_cat_limits = [datareg[datareg.I_cat == _].I.max() for _ in range(11)]
D_cat_limits = [datareg[datareg.D_cat == _].D.max() for _ in range(11)]

I_cat_national_limits = [datareg[datareg.I_cat_national == _].I_national.max() for _ in range(11)]
D_cat_national_limits = [datareg[datareg.D_cat_national == _].D_national.max() for _ in range(11)]

# Alice's formula 
# [ve +  (1-ve)*(1-p(die))] *c(v1) – (1-p(die))*c(v0)

for (district, _, N_district, _, IFR_sero, _) in district_IFR.iloc[2:3].itertuples():
    # evaluate novax first 

    for (vax_pct_annual_goal, vax_effectiveness) in product(
        (0.25, 0.50),
        (0.50, 0.70, 1.00)
    ):
        pass 

It = pd.read_csv("data/full_sims/It_TN_Chennai_novaccination.csv")
Dt = pd.read_csv("data/full_sims/Dt_TN_Chennai_novaccination.csv")

It = It.rename(columns = {"Unnamed: 0": "t"})[["t"] + list(map(str, range(10)))]
It["month"] = It.t.apply(lambda _: simulation_start + pd.Timedelta(_, "days")).dt.month

Dx = pd.read_csv("data/compartment_counts/Dx_TN_Chennai_novaccination.csv").drop(columns = ["Unnamed: 0"]) 

def I_to_cat(I, N):
    return 10 - (I/N).apply(lambda x: next((i for i in reversed(range(11)) if x < I_cat_limits[i]), 0))

It_cat = 10 - (It["0"]/N_district).apply(lambda x: next((i for i in reversed(range(11)) if x < I_cat_limits[i]), 0))
Dt_cat = 10 - (Dt["0"]/N_district).apply(lambda x: next((i for i in reversed(range(11)) if x < D_cat_limits[i]), 0))

def date_to_I_cat_national(m_Y):
    if m_Y == "01_2021":
        return 10
    return 0

def predict(district, It, Dt):
    district_num  = district_codes[district]
    district_code = str(district_num) + "b" if district_num == 92 else str(district_num)
    rchat0 = coeffs.loc["_cons"].value + coeffs.loc[f"{district_code}.districtnum"].value

    rchats = dict()
    for sim in range(1):
        monthly = pd.DataFrame({
            "date": [simulation_start + pd.Timedelta(_, "days") for _ in range(len(It)-1)],
            "I":    It[str(sim)].diff().dropna().clip(0),
            "D":    Dt[str(sim)].diff().dropna().clip(0),
        }).assign(m_Y = lambda _:_.date.dt.strftime("%m_%Y"))\
        .groupby("m_Y")\
        [["I", "D"]]\
        .mean()\
        .reset_index()\
        .assign(
            I_cat          = lambda _:10 - (_.I/N_district).apply(lambda x: next((i for i in reversed(range(11)) if x < I_cat_limits[i]), 0)),
            D_cat          = lambda _:10 - (_.D/N_district).apply(lambda x: next((i for i in reversed(range(11)) if x < D_cat_limits[i]), 0)),
            I_cat_national = lambda _:_.m_Y.apply(date_to_I_cat_national)
        )
        rchats[sim] = rchat0
        for (_, mY, I_cat, D_cat, I_cat_national) in monthly[["m_Y", "I_cat", "D_cat", "I_cat_national"]].itertuples():
            m = mY.split("_")[0].strip("0")
            rchats[sim] += (
                coeffs.loc[(m + 'b' if m == '1' else m) + ".month"].value + 
                coeffs.loc[(str(I_cat) + 'b' if I_cat == 0 else str(I_cat)) + ".I_cat"].value + 
                coeffs.loc[(str(D_cat) + 'b' if D_cat == 0 else str(D_cat)) + ".D_cat"].value + 
                coeffs.loc[(str(I_cat_national) + 'b' if I_cat_national == 0 else str(I_cat_national)) + ".I_cat_national"].value 
            )
        return rchats[sim]

def prob_death(Dx, N_district):
    return (Dx.iloc[-1] - Dx.iloc[1])/split_by_age(N_district)

def vax_weight(ve, Dx, N_district):
    return ve + (1 - ve)*(1 - prob_death(Dx, N_district))

def WTP(district, ve, It_v, Dt_v, Dx_v, It_nv, Dt_nv, Dx_nv):
    N_district = district_populations[district]
    rchat_v  = predict(district, It_v,  Dt_v)
    rchat_nv = predict(district, It_nv, Dt_nv)
    if district == "Kanyakumari":
        district = "Kanniyakumari"
    cons_v  = (1 + rchat_v)  * pcons.loc[district]
    cons_nv = (1 + rchat_nv) * pcons.loc[district]
    if len(It_nv) > len(It_v):
        cons_v += pcons.loc[district] * (len(It_nv) - len(It_v))//30
    return vax_weight(ve, Dx_v, N_district).values * cons_v.values - (1 - prob_death(Dx_nv, N_district)).values * cons_nv.values

def get_WTP(district):
    ve  = 0.7 
    It_v  = pd.read_csv(f"data/full_sims/It_TN_{district}_mortalityprioritized_ve70_annualgoal50_Rt_threshold0.2.csv")
    Dt_v  = pd.read_csv(f"data/full_sims/Dt_TN_{district}_mortalityprioritized_ve70_annualgoal50_Rt_threshold0.2.csv")
    Dx_v  = pd.read_csv(f"data/compartment_counts/Dx_TN_{district}_mortalityprioritized_ve70_annualgoal50_Rt_threshold0.2.csv").drop(columns = ["Unnamed: 0"])
    
    It_nv = pd.read_csv(f"data/full_sims/It_TN_{district}_novaccination.csv")
    Dt_nv = pd.read_csv(f"data/full_sims/Dt_TN_{district}_novaccination.csv")
    Dx_nv = pd.read_csv(f"data/compartment_counts/Dx_TN_{district}_novaccination.csv").drop(columns = ["Unnamed: 0"])
    
    print(" & ".join([district] + list(map(str, WTP(district, ve, It_v, Dt_v, Dx_v, It_nv, Dt_nv, Dx_nv).round(2)))) + " \\\\ ")

for district in district_codes.keys():
    try:
        get_WTP(district)
    except:
        pass