from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import poisson


class SIR():
    """ stochastic SIR model with external introductions """
    def __init__(self, 
        name:                str,           # name of unit
        population:          int,           # unit population
        dT0:        Optional[int]  = None,  # last change in cases 
        Rt0:                 float = 1.9,   # initial reproductive rate,
        I0:                  int   = 0,     # initial infected, None -> Poisson random intro 
        R0:                  int   = 0,     # initial recovered
        D0:                  int   = 0,     # initial dead
        infectious_period:   int   = 5,     # how long disease is communicable in days 
        introduction_rate:   float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:           float = 0.02,  # I -> D transition probability 
        mobility:            float = 0,     # percentage of total population migrating out at each timestep 
        upper_CI:            float = 0.0,   # initial upper confidence interval 
        lower_CI:            float = 0.0,   # initial lower confidence interval 
        CI:                  float = 0.95,  # confidence interval
        random_seed: Optional[int] = None   # random seed 
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
        self.S  = [population - I0 - R0 - D0]
        self.I  = [I0] 
        self.R  = [R0]
        self.D  = [D0]
        self.N  = [population - D0] # total population = S + I + R 
        self.beta = [Rt0 * self.gamma] # initial contact rate 
        self.total_cases = [I0] # total cases 
        self.upper_CI = [upper_CI]
        self.lower_CI = [lower_CI]

        if not random_seed: random_seed = 0
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
        self.N.append(N)
        self.beta.append(beta)
        self.dT.append(num_cases)
        self.total_cases.append(I + R + D)

    def run(self, days: int):
        for _ in range(days):
            self.forward_epi_step()
        return self

    def __repr__(self) -> str:
        return f"[{self.name}]"

class NetworkedSIR():
    """ implements cross-geography interactions between a set of SIR models """
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

    def aggregate(self, curves: Union[Sequence[str], str] = ["Rt", "b", "S", "I", "R", "D", "P", "beta"]) -> Union[Dict[str, Sequence[float]], Sequence[float]]:
        if isinstance(curves, str):
            single_curve = curves
            curves = [curves]
        else: 
            single_curve = False
        aggs = { 
            curve: list(map(sum, zip(*(unit.__getattribute__(curve) for unit in self.units))))
            for curve in curves
        }

        if single_curve:
            return aggs[single_curve]
        return aggs

class SEIR():
    """ stochastic SEIR model with external introductions """

class AR1():
    """ first-order autoregressive model with white noise """

class MigrationSpikeModel(NetworkedSIR):
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
