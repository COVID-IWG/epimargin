from pathlib import Path
from typing import Tuple

import adaptive.plots as plt
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.model      import Model
from adaptive.policy     import simulate_PID_controller
from adaptive.smoothing  import box_filter_local
from pandas import DataFrame

normalize_numeric_strings = lambda n: lambda _: str(int(_)).zfill(n)

def load_us_metros(fips_data_file: Path, population_file: Path, case_timeseries_file: Path, end_date: str = "June 01, 2020") -> Tuple[DataFrame, DataFrame, DataFrame]:
    fips_data = pd.read_csv(fips_data_file, skiprows = 2, skipfooter = 3, usecols = [0, 3, 7, 8, 9, 10], engine = "python").dropna()
    fips_data["fips"] = fips_data["FIPS State Code"].apply(normalize_numeric_strings(2)) + fips_data["FIPS County Code"].apply(normalize_numeric_strings(3))
    fips_data["cbsa"] = fips_data["CBSA Code"].apply(normalize_numeric_strings(5))

    population = pd.read_csv(population_file, names = ["cbsa", "population"])
    population.cbsa = population.cbsa.apply(normalize_numeric_strings(5))
    population.set_index("cbsa", inplace = True)

    case_timeseries = pd.read_csv(case_timeseries_file, usecols = ["date", "fips", "cases"]).dropna()
    case_timeseries.fips = case_timeseries.fips.apply(normalize_numeric_strings(5))
    case_timeseries.set_index(["fips", "date"], inplace = True)

    return (fips_data, population, case_timeseries)

def load_india_states():
    pass 


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
    county_population_file = data_path/"MSA_Population_from_API.csv"  # no idea where luis got this but seems to match up with https://hub.arcgis.com/datasets/4d29eb6f07e94b669c0b90c2aa267100_0/data
    county_timeseries_file = data_path/"us-counties.csv"              # https://github.com/nytimes/covid-19-data/blob/master/live/us-counties.csv
    
    end_date = "2020-05-26"
    (metros, pops, cases) = load_us_metros(fips_data_file, county_population_file, county_timeseries_file)
    cases = cases[cases.index.get_level_values(1) < end_date]

    # run estimation for metro areas
    cities = ["Atlanta", "New York", "Los Angeles", "Miami-"]
    cbsa_mapping = metros[["cbsa", "CBSA Title"]].drop_duplicates().set_index("CBSA Title").to_dict()["cbsa"] 
    for city in cities: 
        # find CBSA code from metro name 
        name, cbsa = next((k, v) for (k, v) in cbsa_mapping.items() if city in k)
        areas, state = name.split(", ")
        filename = areas.lower().replace(" ", "_") + state
        print(name, filename)
        
        # aggregate case time series for metro 
        cbsa_cases = cases.loc[metros[metros.cbsa == cbsa].fips.values]\
            .unstack().fillna(0)\
            .stack(0).sum(axis = 0)

        # get population: 
        pop = pops.loc[cbsa][0]

        (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) \
            = analytical_MPVS(cbsa_cases, CI = CI, smoothing = smoothing, totals=True) 

        dates         = [pd.Timestamp(_).to_pydatetime().date() for _ in dates]
        anomaly_dates = [pd.Timestamp(_).to_pydatetime().date() for _ in anomaly_dates]

        # plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin = 0, ymax = 5, yaxis_colors = False)\
        #     .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
        #     .xlabel("date")\
        #     .ylabel("$R_t$")\
        #     .show()
        
        model = lambda: Model.single_unit(name = name, RR0 = Rt_pred[-1], population = pop, infectious_period = infectious_period, 
            I0 = T_pred[-1], lower_CI = T_CI_lower[-1], upper_CI = T_CI_upper[-1], random_seed = 33)

        forward_pred_period = 9
        t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(forward_pred_period +1)]
        current = model().run(forward_pred_period)
        target  = simulate_PID_controller(model(), 0, forward_pred_period)

        plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, 
            prediction_ts = [
                (current[0].delta_T[1:], current[0].lower_CI[1:], current[0].upper_CI[1:], "orange", r"projection with current $R_t$"),
                (target[0].delta_T[1:],  target[0].lower_CI[1:],  target[0].upper_CI[1:],  "green",  r"projection with $R_t \rightarrow 0.9$")
            ])\
            .adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95)\
            .xlabel("date")\
            .ylabel("cases")\
            .show()

    # run estimation for IL counties - Kankakee, Winnebago, Rock Island, indexed by their CBSA title
    for county in ["Kankakee", "Rockford", "Davenport-Moline-Rock Island"]:
        print(county)
        print([(k, v) for (k, v) in cbsa_mapping.items() if county in k])
        print()
        # name, cbsa = next()
        # print(name, cbsa)