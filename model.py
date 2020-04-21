from typing import Sequence

import numpy as np


def poisson(rate):
    return np.random.poisson(rate, 1)[0]

class ModelUnit():
    def __init__(self, 
        name:              str,           # name of unit
        population:        int,           # unit population
        infectious_period: int   = 5,     # how long disease is communicable in days 
        introduction_rate: float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:         float = 0.02,  # I -> D transition probability 
        mobility:          float = 0.001,  # percentage of total population migrating out at each timestep 
        RR0:               float = 1.9):  # initial reproductive rate 
        
        # save params 
        self.name  = name 
        self.pop0  = population
        self.gamma = 1.0/infectious_period
        self.ll    = introduction_rate
        self.m     = mortality
        self.mu    = mobility
        self.RR0   = RR0
        
        # state and delta vectors 
        I0 = poisson(self.ll) # initial number of cases 
        self.RR = [RR0]
        self.b  = [np.exp(self.gamma * (RR0 - 1.0))]
        self.S  = [population]
        self.I  = [I0] 
        self.R  = [0]
        self.D  = [0]
        self.P  = [population] # total population = S + I + R 
        self.total_cases = [I0] # total cases 
        self.delta_T = [I0] # case change rate, initialized with the first introduction, if any
        # self.delta_D = [0]
        # self.delta_R = [0]

    # period 1: inter-state migratory transmission
    def migration_step(self) -> int:
        # note: update state *in place* since we consider it the same time period 
        outflux = poisson(self.mu * self.I[-1])
        new_I = self.I[-1] - outflux
        if new_I < 0: new_i = 0
        self.I[-1]  = new_I
        self.P[-1] -= outflux
        return outflux

    # period 2: intra-state community transmission
    def forward_epi_step(self, delta_B: int): 
        # get previous state 
        S, I, R, D, P = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.P))

        # update state 
        RR = self.RR0 * float(S)/float(P)
        b  = np.exp(self.gamma * (RR - 1))

        rate_T    = (delta_B + b * (self.delta_T[-1] - delta_B) + self.gamma * RR * delta_B)
        num_cases = poisson(rate_T)

        I += num_cases
        S -= num_cases

        rate_D    = self.m * self.gamma * I
        num_dead  = poisson(rate_D)
        D        += num_dead

        rate_R    = (1 - self.m) * self.gamma * I 
        num_recov = poisson(rate_R)
        R        += num_recov

        I -= (num_dead + num_recov)
        
        if S < 0: S = 0
        if I < 0: I = 0
        if D < 0: D = 0

        # update state vectors 
        self.RR.append(RR)
        self.b.append(b)
        self.S.append(S)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.P.append(P)
        self.delta_T.append(num_cases)
        self.total_cases.append(I + R + D)

    def __repr__(self) -> str:
        return f"ModelUnit[{self.name}]"

class Model():
    def __init__(self, num_days: int, units: Sequence[ModelUnit], migrations: np.matrix):
        self.num_days   = num_days
        self.units      = units
        self.migrations = migrations

    def tick(self):
        # run migration step 
        outflux       = np.array([unit.migration_step() for unit in self.units])
        transmissions = np.ceil(self.migrations * outflux[None, :]).sum(axis = 1)
        
        # now run forward epidemiological model 
        for (unit, tmx) in zip(self.units, transmissions):
            unit.forward_epi_step(tmx)

    def run(self):
        for _ in range(self.num_days):
            self.tick()
        return self 
