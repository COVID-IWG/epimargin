
import numpy as np
import matplotlib.pyplot as plt

def Introductions(ll,k): # lambda is the average number per unit time, k is the number of times
# generates the time series of case introductions, not community transmission
    DeltaB = np.random.poisson(ll,k) # simplest model of introductions.
    return DeltaB

pop = 1000000 #population size
ndays = 200 # number of days
rintroductions = 0.5 # number of infectious introduced in the population per day

infperiod = 5 # 5 days.
gamma = 1./infperiod
m = 0.02 # 2 percent case mortality

DeltaB = Introductions(1, ndays)

RR=np.ones(ndays)
b=[]
for i in range(len(RR)):
    RR[i]=1.5
    b.append( np.exp( gamma*(RR[i]-1.) ) )

S=pop
DeltaT=[]
DeltaT.append(DeltaB[0]) # total cases, initialized with the first introduction, if any

DeltaD=[]
DeltaD.append(0.)

DeltaR=[]
DeltaR.append(0.)

time=np.arange(ndays)

D=0
R=0
I=DeltaB[0]

Susceptibles=[]
Susceptibles.append(S)
Infectious=[]
Infectious.append(I)
Dead=[]
Dead.append(D)
Recovered=[]
Recovered.append(R)

KP=0.05
KI=.5
KD=0.0
Dt=0.1
integral=0.
u=0
error0=RR[0]
Rtarget=0.5
for i in range(1,ndays):
    
    RR[i]=RR[i]*float(S)/float(pop) -u
    b[i]=np.exp( gamma*(RR[i]-1.) )
    error=RR[i]-Rtarget
    
    integral = integral + error * Dt
    derivative = (error - error0) / Dt
    
    u = KP*error + KI*integral + KD*derivative
    error0=error
    print(RR[i],u,error)
    
    rateT=(DeltaB[i]+b[i]*(DeltaT[i-1] - DeltaB[i])+ gamma*RR[i]*DeltaB[i])
    Ncases=np.random.poisson(rateT,1)[0]
    DeltaT.append(Ncases)
    
    I+=Ncases
    S-=Ncases
    
    rateD=gamma*m*I
    Ndead=np.random.poisson(rateD,1)[0]
    #DeltaD.append(Ndead)
    D+=Ndead
    Dead.append(D)
    
    rateR=(1.-m)*gamma*I
    Nrecovered=np.random.poisson(rateR,1)[0]
    #DeltaR.append(Nrecovered)
    R+=Nrecovered
    Recovered.append(R)
    
    I-=(Ndead+Nrecovered)
    
    if (I<0): I=0
    if (S<0): S=0
    Susceptibles.append(S)
    Infectious.append(I)

#plt.semilogy(time,Susceptibles,'g-',lw=5,alpha=0.5)

#plt.semilogy(time,Infectious,'b-',lw=5,alpha=0.5)
#plt.semilogy(time,Dead,'r-',lw=5,alpha=0.5)
#plt.semilogy(time,Recovered,'k-',lw=5,alpha=0.5)

plt.plot(time,Infectious,'b-',lw=5,alpha=0.5)
plt.plot(time,Dead,'r-',lw=5,alpha=0.5)
plt.plot(time,Recovered,'k-',lw=5,alpha=0.5)

plt.xlabel('Number of Days',fontsize=20)
plt.ylabel('Susceptible, Recovered, Infectious, Dead',fontsize=14)
plt.tight_layout()
plt.show()






