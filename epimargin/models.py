from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union, List

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import poisson, binom
from .utils import normalize, fillna

def _max(*args):
    try: 
        return max(*args)
    except ValueError: 
        return 

"""
Stochastic epidemiological models for forward simulation.
"""

class SIR():
    """ stochastic SIR compartmental model with external introductions """
    def __init__(self, 
        name:                str,           # name of unit
        population:          int,           # unit population
        dT0:        Optional[int]  = None,  # last change in cases, None -> Poisson random intro 
        Rt0:                 float = 1.9,   # initial reproductive rate,
        I0:                  int   = 0,     # initial infected
        R0:                  int   = 0,     # initial recovered
        D0:                  int   = 0,     # initial dead
        S0:         Optional[int]  = None,  # initial susceptibles, calculated from N, I, R, D if not provided
        infectious_period:   int   = 5,     # how long disease is communicable in days 
        introduction_rate:   float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:           float = 0.02,  # I -> D transition probability 
        mobility:            float = 0,     # percentage of total population migrating out at each timestep 
        upper_CI:            float = 0.0,   # initial upper confidence interval for new case counts
        lower_CI:            float = 0.0,   # initial lower confidence interval for new case counts
        CI:                  float = 0.95,  # confidence interval
        random_seed:         int   = 0      # random seed 
        ):
        
        # save params 
        self.name  = name 
        self.pop0  = population
        self.gamma = 1.0/infectious_period
        self.ll    = introduction_rate
        self.m     = mortality
        self.mu    = mobility
        self.Rt0   = Rt0
        self.CI    = CI 

        # state and delta vectors 
        if dT0 is None:
            dT0 = np.random.poisson(self.ll) # initial number of new cases 
        self.dT = [dT0] # case change rate, initialized with the first introduction, if any
        self.Rt = [Rt0]
        self.b  = [np.exp(self.gamma * (Rt0 - 1.0))]
        self.S  = [S0 if S0 is not None else population - R0 - D0 - I0]
        self.I  = [I0] 
        self.R  = [R0]
        self.D  = [D0]
        self.dR = [0]
        self.dD = [0]
        self.N  = [population - D0] # total population = S + I + R 
        self.beta = [Rt0 * self.gamma] # initial contact rate 
        self.total_cases = [I0] # total cases 
        self.upper_CI = [upper_CI]
        self.lower_CI = [lower_CI]

        np.random.seed(random_seed)

    # period 1: inter-state migratory transmission
    def migration_step(self) -> int:
        # note: update state *in place* since we consider it the same time period 
        outflux = np.random.poisson(self.mu * self.I[-1])
        new_I = self.I[-1] - outflux
        if new_I < 0: new_I = 0
        self.I[-1]  = new_I
        self.N[-1] -= outflux
        return outflux

    # period 2: intra-state community transmission
    def forward_epi_step(self, dB: int = 0): 
        # get previous state 
        S, I, R, D, N = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.N))

        # update state 
        Rt = self.Rt0 * float(S)/float(N)
        b  = np.exp(self.gamma * (Rt - 1))

        rate_T    = max(0, self.b[-1] * self.dT[-1] + (1 - self.b[-1] + self.gamma * self.b[-1] * self.Rt[-1])*dB)
        num_cases = poisson.rvs(rate_T)
        self.upper_CI.append(poisson.ppf(self.CI,     rate_T))
        self.lower_CI.append(poisson.ppf(1 - self.CI, rate_T))

        I += num_cases
        S -= num_cases

        rate_D    = self.m * self.gamma * I
        num_dead  = poisson.rvs(rate_D)
        D        += num_dead

        rate_R    = (1 - self.m) * self.gamma * I 
        num_recov = poisson.rvs(rate_R)
        R        += num_recov

        I -= (num_dead + num_recov)
        
        if S < 0: S = 0
        if I < 0: I = 0
        if D < 0: D = 0

        N = S + I + R
        beta = (num_cases * N)/(b * S * I)

        # update state vectors 
        self.Rt.append(Rt)
        self.b.append(b)
        self.S.append(S)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.dR.append(num_recov)
        self.dD.append(num_dead)
        self.N.append(N)
        self.beta.append(beta)
        self.dT.append(num_cases)
        self.total_cases.append(I + R + D)
    
    # parallel poisson draws for infection
    def parallel_forward_epi_step(self, dB: int = 0, num_sims = 10000): 
        # get previous state 
        S, I, R, D, N = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.N))

        # update state 
        Rt = self.Rt0 * S/N
        b  = np.exp(self.gamma * (Rt - 1))

        rate_T    = (self.b[-1] * self.dT[-1]).clip(0)
        num_cases = poisson.rvs(rate_T, size = num_sims)
        self.upper_CI.append(poisson.ppf(self.CI,     rate_T))
        self.lower_CI.append(poisson.ppf(1 - self.CI, rate_T))

        I += num_cases
        S -= num_cases

        rate_D    = self.m * self.gamma * I
        num_dead  = poisson.rvs(rate_D, size = num_sims)
        D        += num_dead

        rate_R    = (1 - self.m) * self.gamma * I 
        num_recov = poisson.rvs(rate_R, size = num_sims)
        R        += num_recov

        I -= (num_dead + num_recov)

        S = S.clip(0)
        I = I.clip(0)
        D = D.clip(0)

        N = S + I + R
        beta = (num_cases * N)/(b * S * I)

        # update state vectors 
        self.Rt.append(Rt)
        self.b.append(b)
        self.S.append(S)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.dR.append(num_recov)
        self.dD.append(num_dead)
        self.N.append(N)
        self.beta.append(beta)
        self.dT.append(num_cases)
        self.total_cases.append(I + R + D)

    # parallel binomial draws for infection
    def parallel_forward_binom_step(self, dB: int = 0, num_sims = 10000): 
        # get previous state 
        S, I, R, D, N = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.N))

        # update state 
        Rt = self.Rt0 * S/N
        p = self.gamma * Rt * I/N

        num_cases = binom.rvs(n = S, p = p, size = num_sims)
        self.upper_CI.append(binom.ppf(self.CI,     n = S, p = p))
        self.lower_CI.append(binom.ppf(1 - self.CI, n = S, p = p))

        I += num_cases
        S -= num_cases

        rate_D    = self.m * self.gamma * I
        num_dead  = poisson.rvs(rate_D, size = num_sims)
        D        += num_dead

        rate_R    = (1 - self.m) * self.gamma * I 
        num_recov = poisson.rvs(rate_R, size = num_sims)
        R        += num_recov

        I -= (num_dead + num_recov)

        S = S.clip(0)
        I = I.clip(0)
        D = D.clip(0)

        N = S + I + R

        # update state vectors 
        self.Rt.append(Rt)
        # self.b.append(b)
        self.S.append(S)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.N.append(N)
        # self.beta.append(beta)
        self.dT.append(num_cases)
        self.total_cases.append(I + R + D)

    def run(self, days: int):
        for _ in range(days):
            self.forward_epi_step()
        return self

    def __repr__(self) -> str:
        return f"[{self.name}]"

class Age_SIRVD():
    """ age-structured compartmental model with a vaccinated class for each age bin 
    note that the underlying parallelizing mechanism is different from that of SIR and NetworkedSIR

    """
    def __init__(self,
        name:                str,           # name of unit
        population:          int,           # unit population
        dT0:        Optional[int]  = None,  # last change in cases, None -> Poisson random intro 
        Rt0:                 float = 1.9,   # initial reproductive rate,
        S0:                  np.array = np.array(0), # initial susceptibles
        I0:                  np.array = np.array(0), # initial infected
        R0:                  np.array = np.array(0), # initial recovered
        D0:                  np.array = np.array(0), # initial dead
        infectious_period:   int   = 5,     # how long disease is communicable in days 
        introduction_rate:   float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:           float = 0.02,  # I -> D transition probability 
        mobility:            float = 0,     # percentage of total population migrating out at each timestep 
        upper_CI:            float = 0.0,   # initial upper confidence interval for new case counts
        lower_CI:            float = 0.0,   # initial lower confidence interval for new case counts
        CI:                  float = 0.95,  # confidence interval
        num_age_bins:        int   = 7,     # number of age bins
        phi:                 float = 0.25,  # proportion of population vaccinated annually 
        ve:                  float = 0.7,   # vaccine effectiveness
        random_seed:         int   = 0      # random seed,  
    ):
        self.name  = name 
        self.pop0  = population
        self.gamma = 1.0/infectious_period
        self.ll    = introduction_rate
        self.m     = mortality
        self.mu    = mobility
        self.Rt0   = Rt0
        self.CI    = CI 

        # state and delta vectors 
        if dT0 is None:
            dT0 = np.random.poisson(self.ll) # initial number of new cases 
        self.dT = [dT0] # case change rate, initialized with the first introduction, if any
        self.Rt = [Rt0]
        self.b  = [np.exp(self.gamma * (Rt0 - 1.0))]
        self.S  = [S0 if S0 is not None else population - R0 - D0 - I0]
        self.I  = [I0] 
        self.R  = [R0]
        self.D  = [D0]
        self.dR = [0]
        self.dD = [0]
        self.N  = [population - D0] # total population = S + I + R 
        self.beta = [Rt0 * self.gamma] # initial contact rate 
        self.total_cases = [I0] # total cases 
        self.upper_CI = [upper_CI]
        self.lower_CI = [lower_CI]

        np.random.seed(random_seed)


        self.N = [S0 + I0 + R0]
        shape = (sims, bins) = S0.shape
        
        self.num_age_bins = num_age_bins
        self.phi  = phi
        self.ve   = ve
        
        self.S    = [S0]

        self.S_vm = [np.zeros(shape)]
        self.S_vn = [np.zeros(shape)]
        
        self.I_vn = [np.zeros(shape)]
        
        self.R_vm = [np.zeros(shape)]
        self.R_vn = [np.zeros(shape)]
        
        self.D_vn = [np.zeros(shape)]

        self.N_vn = [np.zeros(shape)] # number vaccinated, ineffective 
        self.N_vm = [np.zeros(shape)] # number vaccinated, immune 

        self.N_v  = [np.zeros(shape)] # total vaccinated
        self.N_nv = [np.zeros(shape)] # total unvaccinated
        self.pi   = [np.zeros(shape)]
        self.q1   = [np.zeros(shape)]
        self.q0   = [np.zeros(shape)]

        self.dT_total = [np.zeros(sims)]
        self.dD_total = [np.zeros(sims)]
        self.dV: List[np.array] = []

        self.rng = np.random.default_rng(random_seed)

    def parallel_forward_epi_step(self, dV: Optional[np.array], num_sims = 10000): 
        """
            in the SIR and NetworkedSIR, the dB is the reservoir introductions; 
            here, dV is a (self.age_bins, num_sims)-sized array of vaccination doses (administered)
        """
        # get previous state 
        S, S_vm, S_vn, I, I_vn, R, R_vm, R_vn, D, D_vn, N, N_vn, N_vm = (_[-1].copy() for _ in 
            (self.S, self.S_vm, self.S_vn, self.I, self.I_vn, self.R, self.R_vm, self.R_vn, self.D, self.D_vn, self.N, self.N_vn, self.N_vm))

        # vaccination occurs here
        dS_vm = (fillna(self.S[-1]/self.N[-1]) * (    self.ve) * dV)
        dS_vn = (fillna(self.S[-1]/self.N[-1]) * (1 - self.ve) * dV)

        dI_vn =  fillna(self.I[-1]/self.N[-1]) * dV
        dR_vm =  fillna(self.R[-1]/self.N[-1]) * dV

        S_vm  = (S_vm + dS_vm).clip(0)
        S_vn  = (S_vn + dS_vn).clip(0)
        S     = (S - (dS_vn + dS_vm)).clip(0)

        I_vn  = (I_vn + dI_vn).clip(0)
        I     = (I    - dI_vn).clip(0)

        R_vm  = (R_vm + dR_vm).clip(0)
        R     = (R    - dR_vm).clip(0)

        S_ratios = normalize(S + S_vn, axis = 1)

        # core epi update with additional bins (infection, death, recovery)
        Rt = self.Rt0 * (S + S_vn).sum(axis = 1)/(N + S_vn + S_vm + I_vn + R_vn + R_vm).sum(axis = 1)
        b  = np.exp(self.gamma * (Rt - 1))

        lambda_T = (self.b[-1] * self.dT[-1])
        dT = np.clip(self.rng.poisson(lambda_T), 0, np.sum(S, axis = 1))
        self.upper_CI.append(poisson.ppf(    self.CI, lambda_T))
        self.lower_CI.append(poisson.ppf(1 - self.CI, lambda_T))

        dS    = fillna(S   /(S+S_vn)) * (S_ratios * dT[:, None])
        dS_vn = fillna(S_vn/(S+S_vn)) * (S_ratios * dT[:, None])

        S    = (S    - dS).clip(0)
        S_vn = (S_vn - dS_vn).clip(0)

        dD    = self.rng.poisson(   self.m  * self.gamma * I   , size = (num_sims, self.num_age_bins))
        dD_vn = self.rng.poisson(   self.m  * self.gamma * I_vn, size = (num_sims, self.num_age_bins))
        dR    = self.rng.poisson((1-self.m) * self.gamma * I   , size = (num_sims, self.num_age_bins))
        dR_vn = self.rng.poisson((1-self.m) * self.gamma * I_vn, size = (num_sims, self.num_age_bins))
        
        dI    = (dS    - (dD    + dR))
        dI_vn = (dS_vn - (dD_vn + dR_vn))

        D    = (D    + dD).clip(0)
        D_vn = (D_vn + dD_vn).clip(0)
        
        R    = (R    + dR).clip(0)
        R_vn = (R_vn + dR_vn).clip(0)
        
        I    = (I    + dI).clip(0)
        I_vn = (I_vn + dI_vn).clip(0)

        N    = S    + I    + R
        N_vn = S_vn + I_vn + R_vn
        N_vm = S_vm +        R_vm 

        # beta = dT[:, None] * N/(b * (S + S_vn) * (I + I_vn))

        # calculate vax policy evaluation metrics 
        N_v  = np.clip((S_vm + S_vn + I_vn + D_vn + R_vn + R_vm), a_min = 0, a_max = self.N[0])
        N_nv = self.N[0] - N_v
        pi   = N_v/self.N[0]

        q1 = np.nan_to_num(1 - (D_vn - self.D_vn[0])/N_v , nan = 0, neginf = 1).clip(0, 1)
        q0 = np.nan_to_num(1 - (D    - self.D   [0])/N_nv, nan = 0, neginf = 1).clip(0, 1)

        # update state vectors 
        self.Rt.append(Rt)
        self.b.append(b)
        self.S.append(S)
        self.S_vm.append(S_vm)
        self.S_vn.append(S_vn)
        self.I.append(I)
        self.I_vn.append(I_vn)
        self.R.append(R)
        self.R_vm.append(R_vm)
        self.R_vn.append(R_vn)
        self.D.append(D)
        self.D_vn.append(D_vn)
        self.dR.append(dR)
        self.dD.append(dD)
        self.N.append(N)
        self.N_vn.append(N_vn)
        self.N_vm.append(N_vm)
        # self.beta.append(beta)
        self.dT.append(dT)
        self.dT_total.append(dT)
        self.dD_total.append((dD + dD_vn).sum(axis = 1))
        self.total_cases.append(I + R + D)
        
        self.N_v.append(N_v)
        self.N_nv.append(N_nv)
        self.pi.append(pi)
        self.q1.append(q1)
        self.q0.append(q0)

        self.dV.append(dV)

class NetworkedSIR():
    """ composition of SIR models implementing cross-geography interactions """
    def __init__(self, units: Sequence[SIR], default_migrations: Optional[np.matrix] = None, random_seed : Optional[int] = None):
        self.units      = units
        self.migrations = default_migrations
        self.names      = {unit.name: unit for unit in units}
        if random_seed is not None:
            np.random.seed(random_seed)

    def __len__(self) -> int:
        return len(self.units)

    def tick(self, migrations: np.matrix):
        # run migration step 
        outflux       = [unit.migration_step() for unit in self.units]
        transmissions = [flux * migrations[i, :].sum() for (i, flux) in enumerate(outflux)]
        
        # now run forward epidemiological model 
        for (unit, tmx) in zip(self.units, transmissions):
            unit.forward_epi_step(tmx)

    def run(self, days: int, migrations: Optional[np.matrix] = None):
        if migrations is None:
            migrations = self.migrations
        for _ in range(days):
            self.tick(migrations)
        return self 

    def __iter__(self) -> Iterator[SIR]:
        return iter(self.units)

    # index units
    def __getitem__(self, idx: Union[str, int]) -> SIR:
        if isinstance(idx, int):
            return self.units[idx]
        return self.names[idx]

    def set_parameters(self, **kwargs):
        for (attr, val) in kwargs.items():
            if callable(val):
                if val.__code__.co_argcount == 1:
                    for unit in self.units:
                        unit.__setattr__(attr, val(unit))
                else: 
                    for (i, unit) in enumerate(self.units):
                        unit.__setattr__(attr, val(i, unit))
            elif isinstance(val, dict):
                for unit in self.units:
                    unit.__setattr__(attr, val[unit.name])
            else: 
                for unit in self.units:
                    unit.__setattr__(attr, val)
        return self 

    def aggregate(self, curves: Union[Sequence[str], str] = ["Rt", "b", "S", "I", "R", "D", "P", "beta"]) -> Dict[str, Sequence[float]]:
        return { 
            curve: list(map(sum, zip(*(unit.__getattribute__(curve) for unit in self.units))))
            for curve in curves
        }

class SEIR():
    """ stochastic SEIR model without external introductions """
    def __init__(self, 
        name:                str,           # name of unit
        population:          int,           # unit population
        dT0:        Optional[int]  = None,  # last change in cases, None -> Poisson random intro 
        Rt0:                 float = 1.9,   # initial reproductive rate,
        E0:                  int   = 0,     # initial exposed
        I0:                  int   = 0,     # initial infected
        R0:                  int   = 0,     # initial recovered
        D0:                  int   = 0,     # initial dead
        infectious_period:   int   = 5,     # how long disease is communicable in days 
        incubation_period:   int   = 5,     # how long the diseas takes to incubate
        introduction_rate:   float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:           float = 0.02,  # I -> D transition probability 
        mobility:            float = 0,     # percentage of total population migrating out at each timestep 
        upper_CI:            float = 0.0,   # initial upper confidence interval for new case counts
        lower_CI:            float = 0.0,   # initial lower confidence interval for new case counts
        CI:                  float = 0.95,  # confidence interval
        random_seed:         int   = 0      # random seed 
        ):
        
        # save params 
        self.name  = name 
        self.pop0  = population
        self.gamma = 1.0/infectious_period
        self.sigma = 1.0/incubation_period
        self.ll    = introduction_rate
        self.m     = mortality
        self.mu    = mobility
        self.Rt0   = Rt0
        self.CI    = CI 

        # state and delta vectors 
        if dT0 is None:
            dT0 = np.random.poisson(self.ll) # initial number of new cases 
        self.dT = [dT0] # case change rate, initialized with the first introduction, if any
        self.Rt = [Rt0]
        self.b  = [np.exp(self.gamma * (Rt0 - 1.0))]
        self.S  = [population - E0 - I0 - R0 - D0]
        self.E  = [E0]
        self.I  = [I0] 
        self.R  = [R0]
        self.D  = [D0]
        self.N  = [population - D0] # total population = S + I + R 
        self.beta = [Rt0 * self.gamma] # initial contact rate 
        self.total_cases = [I0] # total cases 
        self.upper_CI = [upper_CI]
        self.lower_CI = [lower_CI]

        np.random.seed(random_seed)
    
    def forward_epi_step(self, dB: int = 0): 
        # get previous state 
        S, E, I, R, D, N = (vector[-1] for vector in (self.S, self.E, self.I, self.R, self.D, self.N))

        # update state 
        Rt = self.Rt0 * float(S)/float(N)
        b  = np.exp(self.gamma * (Rt - 1))

        rate_T    = max(0, self.b[-1] * self.dT[-1])
        num_cases = poisson.rvs(rate_T)
        self.upper_CI.append(poisson.ppf(self.CI,     rate_T))
        self.lower_CI.append(poisson.ppf(1 - self.CI, rate_T))

        E += num_cases
        S -= num_cases

        rate_I    = self.sigma * E
        num_inf   = poisson.rvs(rate_I)

        E -= num_inf 
        I += num_inf

        rate_D    = self.m * self.gamma * I
        num_dead  = poisson.rvs(rate_D)
        D        += num_dead

        rate_R    = (1 - self.m) * self.gamma * I 
        num_recov = poisson.rvs(rate_R)
        R        += num_recov

        I -= (num_dead + num_recov)
        
        if S < 0: S = 0
        if E < 0: E = 0
        if I < 0: I = 0
        if R < 0: R = 0
        if D < 0: D = 0

        N = S + E + I + R
        beta = (num_cases * N)/(b * S * I)

        # update state vectors 
        self.Rt.append(Rt)
        self.b.append(b)
        self.S.append(S)
        self.E.append(E)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.N.append(N)
        self.beta.append(beta)
        self.dT.append(num_cases)
        self.total_cases.append(E + I + R + D)

class AR1():
    """ first-order autoregressive model with white noise """
    def __init__(self, phi: float = 1.01, sigma: float = 1, I0: int = 10, random_seed: int = 0):
        self.phi = phi 
        self.sigma = sigma
        self.I = [I0]
        np.random.seed(random_seed)
    
    def set_parameters(self, **kwargs):
        if "phi"   in kwargs: self.phi   = kwargs["phi"]
        if "sigma" in kwargs: self.sigma = kwargs["sigma"]
        return self 

    def run(self, days: int):
        for _ in range(days):
            self.I.append(self.phi * self.I[-1] + np.random.normal(scale = self.sigma))
        return self 

class MigrationSpikeModel(NetworkedSIR):
    """ networked SIR model simulating a population influx at a given time """
    def __init__(self, units: Sequence[SIR], introduction_time: Sequence[int], migratory_influx: Dict[str, int], default_migrations: Optional[np.matrix] = None, random_seed: Optional[int] = None):
        self.counter = 0
        self.migratory_influx  = migratory_influx 
        self.introduction_time = introduction_time
        super().__init__(units, default_migrations, random_seed)

    def tick(self, migrations: np.matrix):
        self.counter += 1
        # run migration step 
        outflux       = [unit.migration_step() for unit in self.units]
        transmissions = [flux * migrations[i, :].sum() for (i, flux) in enumerate(outflux)]
        
        # now run forward epidemiological model, and add spike at intro time 
        if self.counter == self.introduction_time:
            for (unit, tmx) in zip(self.units, transmissions):
                unit.forward_epi_step(tmx + self.migratory_influx[unit.name])
        else: 
            for (unit, tmx) in zip(self.units, transmissions):
                unit.forward_epi_step(tmx)

def gravity_matrix(gdf_path: Path, population_path: Path) -> Tuple[Sequence[str], Sequence[float], np.matrix]:
    gdf = gpd.read_file(gdf_path)
    districts = [d.upper() for d in gdf.district.values]

    pop_df = pd.read_csv(population_path)

    # population count is numeric in Maharashtra data and a string in other data - converting to numeric
    if pop_df["Population(2011 census)"].dtype == object:
        pop_df["Population(2011 census)"] = pop_df["Population(2011 census)"].str.replace(",","").apply(float)

    population_mapping = {k.replace("-", " ").upper(): v for (k, v) in zip(pop_df["Name"], pop_df["Population(2011 census)"])}
    populations = [population_mapping[district.upper()] for district in districts]

    centroids = [list(pt.coords)[0] for pt in gdf.centroid]
    P = distance_matrix(centroids, centroids)
    P[P != 0] = P[P != 0] ** -1.0 
    P *= np.array(populations)[:, None]
    P /= P.sum(axis = 0)

    return (districts, populations, P)
