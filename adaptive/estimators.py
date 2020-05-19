import pandas as p
from statsmodels.regression.rolling import RollingOLS


def rollingOLS(totals: pd.DataFrame, window: int = 3, infectious_period: float = 4.5) -> pd.DataFrame:
    # run rolling regressions and get parameters
    model   = RollingOLS.from_formula(formula = "logdelta ~ time", window = window, data = totals)
    rolling = model.fit(method = "lstsq")
    
    growthrates = rolling.params.join(rolling.bse, rsuffix="_stderr")
    growthrates["rsq"] = rolling.rsquared
    growthrates.rename(lambda s: s.replace("time", "gradient").replace("const", "intercept"), axis = 1, inplace = True)

    # calculate growth rates
    growthrates["egrowthrateM"] = growthrates.gradient + 2 * growthrates.gradient_stderr
    growthrates["egrowthratem"] = growthrates.gradient - 2 * growthrates.gradient_stderr
    growthrates["R"]            = growthrates.gradient * infectious_period + 1
    growthrates["RM"]           = growthrates.gradient + 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["Rm"]           = growthrates.gradient - 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["date"]         = growthrates.index
    growthrates["days"]         = totals.time

    return growthrates