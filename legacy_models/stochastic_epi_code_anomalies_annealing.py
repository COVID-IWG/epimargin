import numpy as np
import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import gamma
from scipy.stats import nbinom
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt


def Introductions(ll,k): # lambda is the average number per unit time, k is the number of times
# generates the time series of case introductions, not community transmission
    DeltaB = np.random.poisson(ll,k) # simplest model of introductions.
    return DeltaB


pop=1000000 #population size
ndays=100 # number of days

infperiod=5 # 5 days.
m=0.02 # 2 percent case mortaliy
ll=5 # number of infectious introduced in the population per day
DeltaB=Introductions(ll,ndays)
norm = colors.Normalize(vmin=1, vmax=2*len(DeltaB))
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 1

RR=np.ones(ndays)
b=[]
for i in range(len(RR)):
    RR[i]=1.9 # change to have a  schedule of RR over time, including reductions  due to interventions.
    b.append( np.exp( (RR[i]-1.)/infperiod ) )

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
TotalCases=[]
TotalCases.append(DeltaB[0])

for i in range(1,ndays):
    
    RR[i]=RR[i]*float(S)/float(pop)
    b[i]=np.exp( (RR[i]-1.)/infperiod )

    rateT=(DeltaB[i]+b[i]*(DeltaT[i-1] - DeltaB[i]+ RR[i]*DeltaB[i]/infperiod))
    print(i,rateT,DeltaT[i-1],DeltaB[i-1])
    
    Ncases=np.random.poisson(rateT,1)[0]
    DeltaT.append(Ncases)
    
    I+=Ncases
    S-=Ncases
    
    rateD=m*I/infperiod
    Ndead=np.random.poisson(rateD,1)[0]
    #DeltaD.append(Ndead)
    D+=Ndead
    Dead.append(D)
    
    rateR=(1.-m)*I/infperiod
    Nrecovered=np.random.poisson(rateR,1)[0]
    #DeltaR.append(Nrecovered)
    R+=Nrecovered
    Recovered.append(R)
    
    I-=(Ndead+Nrecovered)
    
    if (I<0): I=0
    if (S<0): S=0
    Susceptibles.append(S)
    Infectious.append(I)
    TotalCases.append(I+R+D)

plt.semilogy(time,Susceptibles,'g-',lw=5,alpha=0.5,label="Susceptibles")
plt.semilogy(time,Infectious,'b-',lw=5,alpha=0.5,label="Infectious")
plt.semilogy(time,Dead,'r-',lw=5,alpha=0.5,label="Deaths")
plt.semilogy(time,Recovered,'k-',lw=5,alpha=0.5,label="Recovered")
plt.xlabel('Number of Days',fontsize=20)
plt.ylabel('Susceptible, Recovered, Infectious, Dead',fontsize=14)
plt.tight_layout()
plt.legend()
plt.show()


######
##### estimation and prediction
#####
#####

plt.clf()

alpha=1. # shape parameter of gamma distribution
beta=2.  # rate parameter of gamma distribution see https://en.wikipedia.org/wiki/Gamma_distribution

#x = np.linspace(gamma.ppf(0.001, a=alpha, scale=1/beta),gamma.ppf(0.999, a=alpha, scale=1/beta), 100)
#y1 = stats.gamma.pdf(x, a=alpha, scale=1./beta)
mean, var, skew, kurt = gamma.stats(a=alpha, scale=1/beta, moments='mvsk')
print('mean, var',alpha,beta,mean,var)
#plt.plot(x, y1, "y-") # this is the initial prior
cl='grey'
mk='o'
edge_color='white'

valpha=[]
vbeta=[]
count=0

pred=[]
pstdM=[]
pstdm=[]
xx=[]
cnt=0
NewCases=[]

for i in range(2,len(Infectious)):
    edge_color, color = sm.to_rgba(cnt), sm.to_rgba(cnt+1)
    cnt += 2
    new_cases=TotalCases[i]-TotalCases[i-1] #-DeltaB[i] +DeltaB[i-1]
    old_new_cases=TotalCases[i-1]-TotalCases[i-2] #-DeltaB[i-1] +DeltaB[i-2]
    
    print('new cases=',new_cases,'old_new_cases=',old_new_cases)
        #print(i,alpha)
    alpha =alpha+new_cases
    beta=beta +old_new_cases
    valpha.append(alpha)
    vbeta.append(beta)
    
    #x = np.linspace(gamma.ppf(0.01, a=alpha, scale=1./beta),gamma.ppf(0.99, a=alpha, scale=1./beta), 100) # this is b
    #xx=1.+infperiod*np.log(x) # this is RR tge reproductive number

    #y1 = stats.gamma.pdf(x, a=alpha, scale=1./beta)
    #mean, var, skew, kurt = gamma.stats(a=alpha, scale=1/beta, moments='mvsk')
    
    #RRest=1.+infperiod*np.log(mean)
    #if (RRest<0.): RRest=0.
    #print('estimated RR=',RRest) # to see the evolution of time parameters, recall that this is b, not RR

    if (new_cases>0. and old_new_cases>0.):
        NewCases.append(new_cases)
    
        r, p = alpha, beta/(old_new_cases+beta)
        mean, var, skew, kurt = nbinom.stats(r, p, moments='mvsk')
        print('mean=',mean,'var=',var)
    
        pred.append(mean)
    
        testciM=nbinom.ppf(0.99, r, p)
        pstdM.append(testciM)
        testcim=nbinom.ppf(0.01, r, p)
        pstdm.append(testcim)
    
        #print("predicted new cases:",int(np.round(mean)),'observed:',new_cases) # to see the evolution of time parameters
        np=p
        nr=r
        flag=0
        while (new_cases>testciM or new_cases<testcim):
            
            print("anomaly",testcim,new_cases,testciM,nr,np)
            #annealing: increase variance so as to encompass anomalous observation: allow Bayesian code to recover
            # mean of negbinomial=r*(1-p)/p  variance= r (1-p)/p**2
            # preserve mean, increase variance--> np=0.8*p (smaller), r= r (np/p)*( (1.-p)/(1.-np) )
            # test anomaly
            
            np=0.5*np # this doubles the variuance, which tends to be small at this stage
            nr= nr*(np/p)*( (1.-p)/(1.-np) )
            
            testciM=nbinom.ppf(0.99, nr, np)
            testcim=nbinom.ppf(0.01, nr, np)
            alpha=nr
            beta=np/(1.-np)*old_new_cases
            flag=1
        else:
            if (flag==1):
                print("anomaly resolved")
                alpha=nr
                beta=np/(1.-np)*old_new_cases

    else:
        NewCases.append(new_cases)
        pred.append(0.)
        pstdM.append(ll)
        pstdm.append(0.)


# time series vs. prediction and anomalies
plt.clf()
x=[]
for i in range(len(NewCases)):
    x.append(i)

plt.fill_between(x,pstdM, pstdm,color='gray', alpha=0.3,label="99% Confidence Interval")

plt.plot(x,pred,'c-',lw=4,alpha=0.8,label="Expected Cases")
plt.plot(x,pstdM,'r-',alpha=0.8)
plt.plot(x,pstdm,'r-',alpha=0.8)
plt.plot(x,NewCases,'bo',ms=8,alpha=0.3,label="Observed Cases")
plt.ylabel("Observed vs Predicted Cases",fontsize=14)
plt.xlabel("Day",fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()









