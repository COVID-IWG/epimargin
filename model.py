from typing import Sequence, Optional 

import matplotlib.pyplot as plt
import matplotlib as mlp 
import numpy as np
import seaborn as sns

sns.set_style("white")
sns.despine()

def poisson(rate):
    return np.random.poisson(rate, 1)[0]

class ModelUnit():
    def __init__(self, 
        name:              str,           # name of unit
        population:        int,           # unit population
        infectious_period: int   = 5,     # how long disease is communicable in days 
        introduction_rate: float = 5.0,   # parameter for new community transmissions (lambda) 
        mortality:         float = 0.02,  # I -> D transition probability 
        mobility:          float = 0.01,  # percentage of total population migrating out at each timestep 
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
        self.RR = np.array([RR0])
        self.b  = np.exp(self.gamma * (self.RR - 1.0))
        self.S  = [population]
        self.I  = [I0] 
        self.R  = [0]
        self.D  = [0]
        self.P  = [population + I0] # total population = S + I + R 
        self.total_cases = [I0] # total cases 
        self.delta_T = [I0] # case change rate, initialized with the first introduction, if any
        # self.delta_D = [0]
        # self.delta_R = [0]

    # period 1: intra-state community transmission
    def tick_internal(self) -> int: 
        # get previous state 
        S, I, R, D, P = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.P))

        # update state 
        RR = self.RR0 * float(S)/float(P)
        b  = np.exp(gamma * (RR - 1))

        new_I     = np.exp(self.gamma * (self.RR - 1.0))
        rate_T    = new_I + b * (self.delta_T[-1] - new_I + RR * gamma * new_I)
        new_cases = poisson(rate_T)

        I += new_cases
        S -= new_cases

        rate_D = self.m * self.gamma * I
        num_dead = poisson(rate_D)
        D += num_dead

        rate_R = (1 - self.m) * self.gamma * I 
        num_recovered = poisson(rate_R)
        R += num_recovered

        I -= (num_dead + num_recovered)
        
        if I < 0: I = 0
        if S < 0: S = 0

        P = S + I + R 
        M_total = round(self.mu * P) # total number of people moving out 
        M_inf   = round(self.mu * I/float(P)) # proportion infected 

        # update state vectors 
        self.RR.append(RR)
        self.b.append(b)
        self.S.append(S)
        self.I.append(I)
        self.R.append(R)
        self.D.append(D)
        self.P.append(P)
        self.delta_T.append(new_cases)
        self.total_cases.append(I + R + D)

    # period 2: migratory transmission
    def tick_external(self, influx):
        # note: update state *in place* since we consider it the same time period 
        pass 

    def __repr__(self) -> str:
        return f"ModelUnit[{self.name}]"

class Model():
    def __init__(self, num_days: int, models: Sequence[ModelUnit], migrations: np.matrix):
        self.num_days   = num_days
        self.models     = models
        self.migrations = migrations

    def tick(self):
        outflux = [m.tick_internal() for m in self.models]
        # resolve outflux
        influxes = []
        for (m, i) in zip(self.models, influxes):
            m.tick_external(i)

    def run(self):
        for _ in range(self.num_days):
            self.tick()
        return self 

    def plot(self, filename: Optional[str] = None) -> mlp.figure.Figure:
        fig, axes = plt.subplots(2, 2)
        if filename: 
            plt.savefig(filename)
        return fig

    def show(self, filename: Optional[str] = None) -> mlp.figure.Figure:
        fig = self.plot(filename)
        plt.show()
        return fig

