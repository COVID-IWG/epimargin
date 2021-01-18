import adaptive.plots as plt
import flat_table
import numpy as np
import pandas as pd
from adaptive.utils import setup
from statsmodels.iolib.summary2 import summary_col
from statsmodels.api import OLS

data, _ = setup()

# case data 
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)


# OWID 
schema = { 
    'Date': "date",
    'Daily change in cumulative total': "daily_tests",
    'Cumulative total': "total_tests",
    'Cumulative total per thousand': "total_per_thousand",
    'Daily change in cumulative total per thousand': "delta_per_thousand",
    '7-day smoothed daily change': "smoothed_delta",
    '7-day smoothed daily change per thousand': "smoothed_delta_per_thousand",
    'Short-term positive rate': "positivity",
    'Short-term tests per case': "tests_per_case"
}

testing = pd.read_csv("data/covid-testing-all-observations.csv", parse_dates=["Date"])
testing = testing[testing["ISO code"] == "IND"]\
            .dropna()\
            [schema.keys()]\
            .rename(columns = schema)
testing["month"]     = testing.date.dt.month

def formula(order: int) -> str: 
    powers = " + ".join(f"np.power(delta_per_thousand, {i + 1})" for i in range(order))
    return f"smoothed_delta ~ -1 + daily_tests + C(month)*({powers})"

model = OLS.from_formula(formula(order = 3), data = testing).fit()
print(summary_col(model, regressor_order = ["daily_tests"], drop_omitted = True))

plt.plot(0.2093 * df["TT"][:, "delta", "tested"],    label = "test-scaled")
plt.plot(         df["TT"][:, "delta", "confirmed"], label = "confirmed")
plt.legend()
plt.show()
