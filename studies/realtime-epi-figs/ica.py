import pandas as pd
from epimargin.utils import cwd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import minmax_scale

data = cwd()/"example_data"
df = pd.read_csv(data/"metro_state_policy_evaluation.csv").dropna()
X = pd.concat([
    df.drop(columns = 
        [col for col in df.columns if col.startswith("RR_")]    + 
        [col for col in df.columns if col.startswith("metro_")] +
        ["metro-state", "date", "state", "state_name", "start_stay_at_home", 
        "end_stay_at_home", "mask_mandate_all", "metro_outbreak_start", "threshold_ind", "cbsa_fips", 
        "new_cases_ts", "daily_confirmed_cases"]),
    # pd.get_dummies(df.state_name, prefix = "state_name")
], axis = 1)

X_normed = minmax_scale(X) 

ica = FastICA(random_state = 0, algorithm = 'deflation', max_iter = 1000)
X_transformed = ica.fit_transform(X_normed)
plt.plot(X_transformed[:, 1], X_transformed[:, 2])  
plt.show()
