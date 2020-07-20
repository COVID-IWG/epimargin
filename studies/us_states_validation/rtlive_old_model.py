import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np
import scipy.stats as sps
import theano
import theano.tensor as tt

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

from datetime import date
from datetime import datetime

from IPython.display import clear_output


def download_patient_data(file_path):
    """ Downloads patient data to data directory
        from: https://stackoverflow.com/questions/16694907/ """
    file_path = os.path.join(file_path, "patients.tar.gz")
    url = "https://github.com/beoutbreakprepared/nCoV2019/raw/master/latest_data/latestdata.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

                    
def get_patient_data(file_path, max_delay=60):
    """ Finds every valid delay between symptom onset and report confirmation
        from the patient line list and returns all the delay samples. """
    
    download_patient_data(file_path)
    file_path = os.path.join(file_path, "patients.tar.gz")
    patients = pd.read_csv(
        file_path,
        parse_dates=False,
        header=1,
        usecols=["country", "date_onset_symptoms", "date_confirmation"],
        low_memory=False,
    )

    patients.columns = ["Country", "Onset", "Confirmed"]
    patients.Country = patients.Country.astype("category")

    # There's an errant reversed date
    patients = patients.replace("01.31.2020", "31.01.2020")
    patients = patients.replace("31.04.2020", "01.05.2020")

    # Only keep if both values are present
    patients = patients.dropna()

    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    patients = patients[is_ten_char(patients.Confirmed) & is_ten_char(patients.Onset)]

    # Convert both to datetimes
    patients.Confirmed = pd.to_datetime(
        patients.Confirmed, format="%d.%m.%Y", errors="coerce"
    )
    patients.Onset = pd.to_datetime(patients.Onset, format="%d.%m.%Y", errors="coerce")

    # Only keep records where confirmed > onset
    patients = patients[patients.Confirmed > patients.Onset]

    # Mexico has many cases that are all confirmed on the same day regardless
    # of onset date, so we filter it out.
    patients = patients[patients.Country.ne("Mexico")]

    # Remove any onset dates from the last two weeks to account for all the
    # people who haven't been confirmed yet.
    #  patients = patients[patients.Onset < patients.Onset.max() - pd.Timedelta(days=14)]

    return patients


def get_delays_from_patient_data(file_path, max_delay=60):
    patients = get_patient_data(file_path=file_path, max_delay=max_delay)
    delays = (patients.Confirmed - patients.Onset).dt.days
    delays = delays.reset_index(drop=True)
    delays = delays[delays.le(max_delay)]
    return delays


def get_delay_distribution(file_path, force_update=False):
    """ Returns the empirical delay distribution between symptom onset and
        confirmed positive case. """

    # The literature suggests roughly 5 days of incubation before becoming
    # having symptoms. See:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/
    INCUBATION_DAYS = 0

    try:
        p_delay_path = os.path.join(file_path, "p_delay.csv")
        p_delay = pd.read_csv(p_delay_path, squeeze=True)
        if force_update==True:
            raise FileNotFoundError
    except FileNotFoundError:
        delays = get_delays_from_patient_data(file_path)
        p_delay = delays.value_counts().sort_index()
        new_range = np.arange(0, p_delay.index.max() + 1)
        p_delay = p_delay.reindex(new_range, fill_value=0)
        p_delay /= p_delay.sum()
        p_delay = (
            pd.Series(np.zeros(INCUBATION_DAYS))
            .append(p_delay, ignore_index=True)
            .rename("p_delay")
        )
        p_delay.to_csv(file_path/"p_delay.csv", index=False)

    return p_delay


def confirmed_to_onset(confirmed, p_delay):

    assert not confirmed.isna().any()
    
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    return onset


def adjust_onset_for_right_censorship(onset, p_delay):
    cumulative_p_delay = p_delay.cumsum()
    
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    
    return adjusted, cumulative_p_delay


class MCMCModel(object):
    
    def __init__(self, region, onset, cumulative_p_delay, window=100):
        
        # Just for identification purposes
        self.region = region
        
        # For the model, we'll only look at the last N
        self.onset = onset.iloc[-window:]
        self.cumulative_p_delay = cumulative_p_delay[-window:]
        
        # Where we store the results
        self.trace = None
        self.trace_index = self.onset.index[1:]

    def run(self, chains=1, tune=3000, draws=1000, target_accept=.95):

        with pm.Model() as model:

            # Random walk magnitude
            step_size = pm.HalfNormal('step_size', sigma=.03)

            # Theta random walk
            theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
            theta_raw_steps = pm.Normal('theta_raw_steps', shape=len(self.onset)-2) * step_size
            theta_raw = tt.concatenate([[theta_raw_init], theta_raw_steps])
            theta = pm.Deterministic('theta', theta_raw.cumsum())

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta/gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            self.trace = pm.sample(
                chains=chains,
                tune=tune,
                draws=draws,
                target_accept=target_accept)
            
            return self


def create_and_run_model(name, state_df, p_delay):
    
    confirmed = state_df['delta_positive'].dropna().rename('confirmed')
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return MCMCModel(name, onset, cumulative_p_delay).run()


def df_from_model(model):
    
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    hpd_95 = pm.stats.hpd(r_t, 0.95)
    hpd_50 = pm.stats.hpd(r_t, 0.5)
    
    idx = pd.MultiIndex.from_product([
            [model.region],
            model.trace_index
        ], names=['state', 'date'])
        
    df = pd.DataFrame(data=np.c_[mean, median, hpd_95, hpd_50], index=idx,
                      columns=['mean', 'median', 'lower_95', 'upper_95', 'lower_50','upper_50'])
    df.reset_index(inplace=True)
    return df
