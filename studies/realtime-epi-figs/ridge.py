import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epimargin.utils import cwd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import minmax_scale
from sklearn.svm import LinearSVC

import seaborn as sns 

data = cwd()/"example_data"
df = pd.read_csv(data/"metro_state_policy_evaluation.csv").dropna()
df["Rt_binarized"] = (df["RR_pred"] >= 1).astype(int)
X = pd.concat([
    df.drop(columns = 
        [col for col in df.columns if col.startswith("RR_")]    + 
        [col for col in df.columns if col.startswith("metro_")] +
        ["metro-state", "date", "state", "state_name", "start_stay_at_home", 
        "end_stay_at_home", "mask_mandate_all", "metro_outbreak_start", "threshold_ind", "cbsa_fips", 
        "new_cases_ts", "daily_confirmed_cases"]),
    # pd.get_dummies(df.state_name, prefix = "state_name")
], axis = 1)

X_normed = minmax_scale(X.drop(columns = ["Rt_binarized"])) 

ridge = Ridge(alpha = 0.1, random_state = 0)
ridge.fit(X = X_normed, y = X["Rt_binarized"])
for _  in sorted(zip(X.columns, ridge.coef_), key = lambda t: np.abs(t[1]), reverse = True):
    print(_)
plt.plot(ridge.coef_, ".")
plt.show()

lasso = Lasso(alpha = 0.001, random_state = 0)
lasso.fit(X = X_normed, y = X["Rt_binarized"])
for _  in sorted(zip(X.columns, lasso.coef_), key = lambda t: np.abs(t[1]), reverse = True):
    print(_)
plt.plot(np.abs(lasso.coef_), ".")
plt.show()

enet = ElasticNet(alpha = 0.001, random_state = 0)
enet.fit(X = X_normed, y = X["Rt_binarized"])
for _  in sorted(zip(X.columns, enet.coef_), key = lambda t: np.abs(t[1]), reverse = True):
    print(_)
plt.plot(np.abs(enet.coef_), ".")
plt.show()

enetcv = ElasticNetCV(l1_ratio = [0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 0.95, 0.99, 1], n_alphas = 10000, random_state = 0)
enetcv.fit(X = X_normed, y = df["RR_pred"])
print(enetcv.l1_ratio_, enetcv.alpha_)
for _  in sorted(zip(X.columns, enetcv.coef_), key = lambda t: np.abs(t[1]), reverse = True):
    print(_)
plt.plot(enetcv.coef_, ".")
plt.gca().set_xticklabels([""] + [_.split("_")[0] for _ in X.columns[:-1]])
plt.show()

enetcv_bin = ElasticNetCV(l1_ratio = [0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 0.95, 0.99, 1], n_alphas = 10000, random_state = 0)
enetcv_bin.fit(X = X_normed, y = X["Rt_binarized"])
print(enetcv_bin.l1_ratio_, enetcv_bin.alpha_)
for _  in sorted(zip(X.columns, enetcv_bin.coef_), key = lambda t: np.abs(t[1]), reverse = True):
    print(_)
plt.plot(enetcv_bin.coef_, ".")
plt.gca().set_xticklabels([""] + [_.split("_")[0] for _ in X.columns[:-1]])
plt.show()

Xn_indexed = pd.concat([
    pd.DataFrame(X_normed, columns = X.columns[:-1]),
    X["Rt_binarized"]
], axis = 1)

sns.scatterplot(data=Xn_indexed, x = "retail_and_recreation_percent_change_from_baseline", y = "grocery_and_pharmacy_percent_change_from_baseline", hue = "Rt_binarized")
plt.show()

# 2D projection
sparse_pca = SparsePCA(n_components = 2, random_state = 0, alpha = 2)
X_scaled = minmax_scale(X)
sparse_pca.fit(X = X_scaled)
print(pd.DataFrame(sparse_pca.components_, columns = [_.replace("_percent_change_from_baseline", "") for _ in X.columns]))
X_tf = sparse_pca.transform(X_scaled)
X_tf_Rt = pd.DataFrame(X_tf, columns = ["X1", "X2"])
X_tf_Rt["Rt"] = X["Rt_binarized"]
sns.scatterplot(data = X_tf_Rt, x = "X1", y = "X2", hue = "Rt_binarized")
plt.show()


ax = fig.add_subplot(111, projection='3d')

svc = LinearSVC(random_state = 0, penalty = "l1", loss = "squared_hinge", dual = False, max_iter = 10000)
svc.fit(X = X_normed, y = X["Rt_binarized"])
