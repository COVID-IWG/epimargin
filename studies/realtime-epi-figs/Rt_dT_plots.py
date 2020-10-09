from pathlib import Path
from typing import Tuple, Dict

import adaptive.plots as plt
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.model      import Model
from adaptive.policy     import simulate_PID_controller
from adaptive.smoothing  import box_filter_local
from pandas import DataFrame

normalize_numeric_strings = lambda n: lambda _: str(int(_)).zfill(n)

def load_metro_data(fips_data_file: Path) -> DataFrame:
    fips_data = pd.read_csv(fips_data_file, skiprows = 2, skipfooter = 3, usecols = [0, 3, 7, 8, 9, 10], engine = "python").dropna()
    fips_data["fips"] = fips_data["FIPS State Code"].apply(normalize_numeric_strings(2)) + fips_data["FIPS County Code"].apply(normalize_numeric_strings(3))
    fips_data["cbsa"] = fips_data["CBSA Code"].apply(normalize_numeric_strings(5))
    return fips_data

def load_us_cases(case_timeseries_file: Path) -> DataFrame:
    case_timeseries = pd.read_csv(case_timeseries_file, usecols = ["date", "fips", "cases"]).dropna()
    case_timeseries.fips = case_timeseries.fips.apply(normalize_numeric_strings(5))
    case_timeseries.set_index(["fips", "date"], inplace = True)
    return case_timeseries

def load_cbsa_population(cbsa_population_file: Path) -> DataFrame:
    population = pd.read_csv(cbsa_population_file, names = ["cbsa", "population"])
    population.cbsa = population.cbsa.apply(normalize_numeric_strings(5))
    population.set_index("cbsa", inplace = True)
    return population

def load_fips_population(fips_population_file: Path) -> Dict[str, int]:
    latest = pd.read_csv(fips_population_file, skiprows=1, skipfooter=5, engine="python")[["Geography", "2019"]]\
        .drop(0)\
        .rename({"Geography": "county", "2019": "population"}, axis = 1)
    latest.county = latest.county.str.replace("^.", "").str.replace(" County, Illinois", "")
    latest.population = latest.population.str.replace(",", "").astype(int)
    return latest.set_index("county").to_dict()["population"]

def load_india_cases(india_case_file: Path) -> DataFrame:
    return pd.read_csv(india_case_file, parse_dates = ["Date"], dayfirst = True, usecols = ["Date", "State/UnionTerritory", "Confirmed"])\
        .rename({"Date": "date", "State/UnionTerritory": "state", "Confirmed": "confirmed"}, axis = 1)

if __name__ == "__main__":
    # set up model details 
    infectious_period = 4.5 
    CI        = 0.99 
    window    = 5
    smoothing = box_filter_local(window)

    # set up data paths
    extension = "tex" # change to png for rasters
    data_path = Path("./example_data")
    # variable                        # filename                      # provenance 
    fips_data_file         = data_path/"MSA_County_2018.csv"          # census bureau 
    cbsa_population_file   = data_path/"MSA_Population_from_API.csv"  # no idea where luis got this but seems to match up with https://hub.arcgis.com/datasets/4d29eb6f07e94b669c0b90c2aa267100_0/data
    fips_population_file   = data_path/"IL_county_pops.csv"           # cleaned up version of https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres-17.xlsx
    county_timeseries_file = data_path/"us-counties.csv"              # https://github.com/nytimes/covid-19-data/blob/master/live/us-counties.csv
    india_cases_file       = data_path/"covid_19_india.csv"           # from kaggle? check with luis on provenance
    
    end_date = "2020-05-26"
    metros    = load_metro_data(fips_data_file)
    cases     = load_us_cases(county_timeseries_file)
    cbsa_pops = load_cbsa_population(cbsa_population_file) 
    fips_pops = load_fips_population(fips_population_file) 
    india     = load_india_cases(india_cases_file)

    cases = cases[cases.index.get_level_values(1) < end_date]

    # run estimation for metro areas
    # cities = ["Atlanta", "New York", "Los Angeles", "Miami-"]
    # cbsa_mapping = metros[["cbsa", "CBSA Title"]].drop_duplicates().set_index("CBSA Title").to_dict()["cbsa"] 
    # for city in cities: 
    #     # find CBSA code from metro name 
    #     name, cbsa = next((k, v) for (k, v) in cbsa_mapping.items() if city in k)
    #     areas, state = name.split(", ")
    #     filename = areas.lower().replace(" ", "_") + state
    #     print(name, filename)
        
    #     # aggregate case time series for metro 
    #     cbsa_cases = cases.loc[metros[metros.cbsa == cbsa].fips.values]\
    #         .unstack().fillna(0)\
    #         .stack(0).sum(axis = 0)

    #     (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) \
    #         = analytical_MPVS(cbsa_cases, CI = CI, smoothing = smoothing, totals=True) 

    #     dates         = [pd.Timestamp(_).to_pydatetime().date() for _ in dates]
    #     anomaly_dates = [pd.Timestamp(_).to_pydatetime().date() for _ in anomaly_dates]

    #     plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin = 0, ymax = 5, yaxis_colors = False)\
    #         .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
    #         .xlabel("date")\
    #         .ylabel("$R_t$")\
    #         .show()
        
    #     model = lambda: Model.single_unit(name = name, RR0 = Rt_pred[-1], population = cbsa_pops.loc[cbsa][0], infectious_period = infectious_period, 
    #         I0 = T_pred[-1], lower_CI = T_CI_lower[-1], upper_CI = T_CI_upper[-1], random_seed = 33)

    #     forward_pred_period = 9
    #     t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(forward_pred_period +1)]
    #     current = model().run(forward_pred_period)
    #     target  = simulate_PID_controller(model(), 0, forward_pred_period)

    #     plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
    #         prediction_ts = [
    #             (current[0].delta_T[1:], current[0].lower_CI[1:], current[0].upper_CI[1:], "orange", r"projection with current $R_t$"),
    #             (target[0].delta_T[1:],  target[0].lower_CI[1:],  target[0].upper_CI[1:],  "green",  r"projection with $R_t \rightarrow 0.9$")
    #         ])\
    #         .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
    #         .xlabel("date")\
    #         .ylabel("cases")\
    #         .show()

    # # run estimation for IL counties - Kankakee, Winnebago, Rock Island, indexed by their CBSA title
    # for (cbsa, county) in [("Kankakee", "Kankakee"), ("Rockford", "Winnebago"), ("Davenport-Moline-Rock Island", "Rock Island")]:
    #     print(cbsa, county)
    #     fips_code = metros[metros["County/County Equivalent"].str.contains(county) & (metros["State Name"] == "Illinois")].fips.iloc[0]
    #     (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) \
    #         = analytical_MPVS(cases.loc[fips_code], CI = CI, smoothing = smoothing, totals=True) 
    #     dates         = [pd.Timestamp(_).to_pydatetime().date() for _ in dates]
    #     anomaly_dates = [pd.Timestamp(_).to_pydatetime().date() for _ in anomaly_dates]
    #     model = lambda: Model.single_unit(name = county, RR0 = Rt_pred[-1], population = fips_pops[county], infectious_period = infectious_period, 
    #         I0 = T_pred[-2], lower_CI = T_CI_lower[-2], upper_CI = T_CI_upper[-2], random_seed = 33)
    #     forward_pred_period = 9
    #     t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(forward_pred_period +1)]
    #     current = model().run(forward_pred_period)
    #     target  = simulate_PID_controller(model(), 0, forward_pred_period)
    #     plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin = 0, ymax = 5, yaxis_colors = False)\
    #         .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
    #         .xlabel("date")\
    #         .ylabel("$R_t$")\
    #         .show()
    #     plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
    #         prediction_ts = [
    #             (current[0].delta_T[1:], current[0].lower_CI[1:], current[0].upper_CI[1:], "orange", r"projection with current $R_t$"),
    #             (target[0].delta_T[1:],  target[0].lower_CI[1:],  target[0].upper_CI[1:],  "green",  r"projection with $R_t \rightarrow 0.9$")
    #         ])\
    #         .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
    #         .xlabel("date")\
    #         .ylabel("cases")\
    #         .show()

    # run Indian states
    for (state, pop) in [("Maharashtra", 112374333), ("Gujarat", 60439692), ("Bihar", 104099452)]:
        print(state)
        (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) \
            = analytical_MPVS(india[india.state == state][["date", "confirmed"]].set_index("date")  , CI = CI, smoothing = smoothing, totals=True) 
        dates         = [pd.Timestamp(_).to_pydatetime().date() for _ in dates]
        anomaly_dates = [pd.Timestamp(_).to_pydatetime().date() for _ in anomaly_dates]
        model = lambda: Model.single_unit(name = state, RR0 = Rt_pred[-1], population = pop, infectious_period = infectious_period, 
            I0 = T_pred[-1], lower_CI = T_CI_lower[-1], upper_CI = T_CI_upper[-1], random_seed = 33)
        forward_pred_period = 9
        t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(forward_pred_period +1)]
        current = model().run(forward_pred_period)
        target  = simulate_PID_controller(model(), 0, forward_pred_period)
        plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin = 0, ymax = 5, yaxis_colors = False)\
            .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
            .xlabel("date")\
            .ylabel("$R_t$")\
            .show()
        plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
            prediction_ts = [
                (current[0].delta_T[1:], current[0].lower_CI[1:], current[0].upper_CI[1:], "orange", r"projection with current $R_t$"),
                (target[0].delta_T[1:],  target[0].lower_CI[1:],  target[0].upper_CI[1:],  "green",  r"projection with $R_t \rightarrow 0.9$")
            ])\
            .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
            .xlabel("date")\
            .ylabel("cases")\
            .show()
