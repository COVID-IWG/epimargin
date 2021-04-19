from pathlib import Path
from warnings import simplefilter

import pandas as pd

from epimargin.etl.covid19india import (data_path, download_data,
                                       get_time_series, load_all_data,
                                       load_statewise_data)
from epimargin.utils import cwd

simplefilter("ignore")

def hydrate_datastore(_):
    root = Path("/tmp")
    data = root/"data"
    figs = root/"figs"

    data.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    # define data versions for api files
    paths = {
        "v3": [data_path(i) for i in (1, 2)],
        "v4": [data_path(i) for i in (3, 4, 5, 6, 7, 8, 9, 10)]
    }

    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    df = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )

    data_recency = str(df["date_announced"].max()).split()[0]
    run_date     = str(pd.Timestamp.now()).split()[0]

    print(f"data_recency: {data_recency}")
    print(f"run_date    : {run_date}")

    df["hash"] = df.apply(lambda x: hash(tuple(x)), axis = 1)
    df["report_date"] = run_date 

    df.to_csv(data/f"hashed_records_{run_date}.csv")
    print(df.tail())
