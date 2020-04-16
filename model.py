from typing import Optional, Sequence, Tuple

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mlp.rcParams['font.sans-serif'] = "PT Sans Regular"
mlp.rcParams['font.family'] = "sans-serif"
sns.set_style("whitegrid")
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
        self.RR = [RR0]
        self.b  = [np.exp(self.gamma * (RR0 - 1.0))]
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
    def tick_internal(self) -> Tuple[int]: 
        # get previous state 
        S, I, R, D, P = (vector[-1] for vector in (self.S, self.I, self.R, self.D, self.P))

        # update state 
        RR = self.RR0 * float(S)/float(P)
        b  = np.exp(self.gamma * (RR - 1))

        new_I     = np.exp(self.gamma * (RR - 1.0))
        rate_T    = new_I + b * (self.delta_T[-1] - new_I + RR * self.gamma * new_I)
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
        M_inf   = round(self.mu * I) # number infected 
        P -= M_total

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

        return (M_total, M_inf)

    # period 2: migratory transmission
    def tick_external(self, total_in: int, infected_in: int):
        # note: update state *in place* since we consider it the same time period 
        self.P[-1] += total_in
        self.I[-1] += infected_in 

    def __repr__(self) -> str:
        return f"ModelUnit[{self.name}]"

class Model():
    def __init__(self, num_days: int, models: Sequence[ModelUnit], migrations: np.matrix):
        self.num_days   = num_days
        self.models     = models
        self.migrations = migrations

    def tick(self):
        outflux = [m.tick_internal() for m in self.models]
        total_out, inf_out = zip(*outflux)
        migrated_out = np.round(total_out * self.migrations)
        migrated_inf = np.round(inf_out   * self.migrations)

        for (i, m) in enumerate(self.models):
            m.tick_external(sum(migrated_out[i, :]), sum(migrated_inf[i, :]))

    def run(self):
        for _ in range(self.num_days):
            self.tick()
        return self 

    def plot(self, filename: Optional[str] = None) -> mlp.figure.Figure:
        fig, axes = plt.subplots(1, 4, sharex = True, sharey = True)
        fig.suptitle('Four-State Toy Example (No Adaptive Controls)')
        t = list(range(self.num_days + 1))
        for (ax, model) in zip(axes.flat, self.models):
            s = ax.semilogy(t, model.S, alpha=0.75, label="Susceptibles")
            i = ax.semilogy(t, model.I, alpha=0.75, label="Infectious", )
            d = ax.semilogy(t, model.D, alpha=0.75, label="Deaths",     )
            r = ax.semilogy(t, model.R, alpha=0.75, label="Recovered",  )
            ax.set(xlabel = "# days", ylabel = "S/I/R/D", title = f"{model.name} (initial pop: {model.pop0})")
            ax.label_outer()
            # for tick in ax.get_xticklabels():
            #     tick.set_fontname("Fira Code Light")
            # for tick in ax.get_yticklabels():
            #     tick.set_fontname("Fira Code Light")
            # ax.tight_layout()
        
        fig.legend([s, i, r, d], labels = ["S", "I", "R", "D"], loc="center right", borderaxespad=0.1)
        plt.subplots_adjust(right=0.85)
        # if filename: 
        #     plt.savefig(filename)
        return fig

    def show(self, filename: Optional[str] = None) -> mlp.figure.Figure:
        fig = self.plot(filename)
        plt.show()
        return fig
