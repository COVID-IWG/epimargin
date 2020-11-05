import pandas as pd

from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data, load_data_v3, load_data_v4, drop_cols_v3, drop_cols_v4
import adaptive.plots as plt
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd

# model details
CI        = 0.95
smoothing = 14

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 18)]
}

for target in paths['v3'] + paths['v4']:
    download_data(data, target)

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)

drop_cols_v3.remove("Age Bracket")
drop_cols_v4.remove("Age Bracket")

cases_v3 = [load_data_v3(data/filepath, drop = drop_cols_v3) for filepath in paths["v3"]]
cases_v4 = [load_data_v4(data/filepath, drop = drop_cols_v4) for filepath in paths["v4"]]
all_cases = pd.concat(cases_v3 + cases_v4)

ka_cases = all_cases.query("detected_state == 'Karnataka'")\
    .dropna(subset = ["age_bracket"])