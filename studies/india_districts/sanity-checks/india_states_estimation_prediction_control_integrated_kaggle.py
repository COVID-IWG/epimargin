import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from matplotlib import cm
from numpy import log as ln
#from scipy.stats import gamma
from scipy.stats import nbinom, poisson

from etl import get_time_series, load_all_data

sns.set(style = "whitegrid", palette = "bright", font = "Fira Code")
sns.despine()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# gg=csv.writer(open('Rt_Indian_States.csv','w'))

#### data input from csv file, updated live from here
# states = [
#     'Andaman And Nicobar Islands',
#     'Andhra Pradesh',
#     'Arunachal Pradesh',
#     'Assam',
#     'Bihar',
#     'Chandigarh',
#     'Chhattisgarh',
#     'Delhi',
#     'Goa',
#     'Gujarat',
#     'Haryana',
#     'Himachal Pradesh',
#     'Jammu And Kashmir',
#     'Jharkhand',
#     'Karnataka',
#     'Kerala',
#     'Ladakh',
#     'Madhya Pradesh',
#     'Maharashtra',
#     'Manipur',
#     'Mizoram',
#     'Odisha',
#     'Puducherry',
#     'Punjab',
#     'Rajasthan',
#     'Tamil Nadu',
#     'Telangana',
#     'Uttar Pradesh',
#     'Uttarakhand',
# ]

states = ["Maharashtra"]

#problem states 'Nagaland', 'Sikkim' because of no cases

infperiod = 5 # length of infectious period, adjust as needed

data = Path("./data")
figs = Path("./figs/comparison/kaggle")

for state in states:
    from scipy.stats import gamma # not sure why this needs to be recalled after each state, but otherwite get a type exception
    
    fig, ax = plt.subplots()

    g=open('./data/covid_19_india.csv', 'r')
    reader=csv.reader(g)

    confirmed=[]
    dead=[]
    recovered=[]
    dates=[]
    day=[]
    
    ii=0
    for row in reader:
        if (row[3]==state and ii >0):
            date_object = datetime.strptime(row[1], '%d/%m/%y').date()
            day.append(float(ii))
            #print(date_object,float(row[8]),float(row[7]),float(row[6]))
            dates.append(date_object)
            recovered.append(float(row[6]))
            dead.append(float(row[7]))
            confirmed.append(float(row[8]))
        ii+=1
    g.close()

    ndays=[]
    for i in range (len(confirmed)):
        if (confirmed[i]>0):
            ndays.append(day[i])
    print(state,'days with cases=',len(ndays),'total cases=',confirmed[-1])
    if (confirmed[-1] < 10.):
        print(f"skipping {state} due to fewer than 10 cases")
        continue  # this skips the Rt analysis for states for which there are <10 total cases

##### estimation and prediction
    dconfirmed=np.diff(confirmed)
    for ii in range(len(dconfirmed)):
        if dconfirmed[ii]<0. : dconfirmed[ii]=0.
    xd=dates[1:]

    plt.plot(xd,dconfirmed,'go',alpha=0.5,markersize=8,label='Reported daily new cases')
    sdays=5
    yy=smooth(dconfirmed,sdays) # smoothing over sdays (number of days) moving window, averages large chunking in reporting in consecutive days
    yy[-2]=(dconfirmed[-4]+dconfirmed[-3]+dconfirmed[-2])/3. # these 2 last lines should not be necesary but the data tend to be initially underreported and also the smoother struggles.
    yy[-1]=(dconfirmed[-3]+dconfirmed[-2]+dconfirmed[-1])/3.


    plt.title(state, fontsize=20, loc="left", fontdict={"family": "Fira Sans", "fontweight": "500"})
    plt.plot(xd, yy, 'b-', lw=2,label='Smoothened case time series')
    plt.ylabel("Observed New Cases",fontsize=16)
    plt.xlabel("Day",fontsize=20)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.legend()
    plt.gcf().set_size_inches(11, 8)
    plt.savefig(figs/f'NewCases_Timeseries_{state}.png', dpi=600, bbox_inches="tight")
    plt.close()
    #plt.show()
    plt.clf()

#lyyy=np.cumsum(lwy)
    TotalCases=np.cumsum(yy) # These are confirmed cases after smoothing: tried also a lowess smoother but was a bit more parameer dependent from place to place.

    alpha=3. # shape parameter of gamma distribution
    beta=2.  # rate parameter of gamma distribution see https://en.wikipedia.org/wiki/Gamma_distribution

    valpha=[]
    vbeta=[]

    pred=[]
    pstdM=[]
    pstdm=[]
    xx=[]
    NewCases=[]

    predR=[]
    pstRRM=[]
    pstRRm=[]

    anomalyday=[]
    anomalypred=[]

    for i in range(2,len(TotalCases)):
        new_cases=float(TotalCases[i]-TotalCases[i-1])
        old_new_cases=float(TotalCases[i-1]-TotalCases[i-2])
        
        # This uses a conjugate prior as a Gamma distribution for b_t, with parameters alpha and beta
        alpha =alpha+new_cases
        beta=beta +old_new_cases
        valpha.append(alpha)
        vbeta.append(beta)
        
        mean = gamma.stats(a=alpha, scale=1/beta, moments='m')
        
        RRest=1.+infperiod*ln(mean)
        if (RRest<0.): RRest=0.
        predR.append(RRest)
        testRRM=1.+infperiod*ln( gamma.ppf(0.99, a=alpha, scale=1./beta) )# these are the boundaries of the 99% confidence interval  for new cases
        if (testRRM <0.): testRRM=0.
        pstRRM.append(testRRM)
        testRRm=1.+infperiod*ln( gamma.ppf(0.01, a=alpha, scale=1./beta) )
        if (testRRm <0.): testRRm=0.
        pstRRm.append(testRRm)
        
        #print('estimated RR=',RRest,testRRm,testRRM) # to see the numbers for the evolution of Rt
        
        if (new_cases==0. or old_new_cases==0.):
            pred.append(0.)
            pstdM.append(10.)
            pstdm.append(0.)
            NewCases.append(0.)
        
        if (new_cases>0. and old_new_cases>0.):
            NewCases.append(new_cases)
            
            # Using a Negative Binomial as the  Posterior Predictor of New Cases, given old one
            # This takes parameters r,p which are functions of new alpha, beta from Gamma
            r, p = alpha, beta/(old_new_cases+beta)
            mean, var, skew, kurt = nbinom.stats(r, p, moments='mvsk')
            
            pred.append(mean) # the expected value of new cases
            testciM=nbinom.ppf(0.99, r, p) # these are the boundaries of the 99% confidence interval  for new cases
            pstdM.append(testciM)
            testcim=nbinom.ppf(0.01, r, p)
            pstdm.append(testcim)
            
            np=p
            nr=r
            flag=0
            
            while (new_cases>testciM or new_cases<testcim):
                if (flag==0):
                    anomalypred.append(new_cases)
                    anomalyday.append(dates[i+1]) # the first new cases are at i=2
                
                #print("anomaly",testcim,new_cases,testciM,nr,np) #New  cases when falling outside the 99% CI
                #annealing: increase variance so as to encompass anomalous observation: allow Bayesian code to recover
                # mean of negbinomial=r*(1-p)/p  variance= r (1-p)/p**2
                # preserve mean, increase variance--> np=0.8*p (smaller), r= r (np/p)*( (1.-p)/(1.-np) )
                # test anomaly
                
                nnp=0.95*np # this doubles the variance, which tends to be small after many Bayesian steps
                nr= nr*(nnp/np)*( (1.-np)/(1.-nnp) ) # this assignement preserves the mean of expected cases
                np=nnp
                mean, var, skew, kurt = nbinom.stats(nr, np, moments='mvsk')
                testciM=nbinom.ppf(0.99, nr, np)
                testcim=nbinom.ppf(0.01, nr, np)
                
                flag=1
            else:
                if (flag==1):
                    alpha=nr  # this updates the R distribution  with the new parameters that enclose the anomaly
                    beta=np/(1.-np)*old_new_cases
                    
                    testciM=nbinom.ppf(0.99, nr, np)
                    testcim=nbinom.ppf(0.01, nr, np)
                    
                    #pstdM=pstdM[:-1] # remove last element and replace by expanded CI for New Cases
                    #pstdm=pstdm[:-1]  # This (commented) in  order to show anomalies, but on
                    #pstdM.append(testciM) # in the parameter update, uncomment and it will plot the actual updated CI
                    #pstdm.append(testcim)
                    
                    
                    # annealing leaves the RR mean unchanged, but we need to adjus its widened CI:
                    testRRM=1.+infperiod*ln( gamma.ppf(0.99, a=alpha, scale=1./beta) )# these are the boundaries of the 99% confidence interval  for new cases
                    if (testRRM <0.): testRRM=0.
                    testRRm=1.+infperiod*ln( gamma.ppf(0.01, a=alpha, scale=1./beta) )
                    if (testRRm <0.): testRRm=0.
                    
                    pstRRM=pstRRM[:-1] # remove last element and replace by expanded CI for RRest
                    pstRRm=pstRRm[:-1]
                    pstRRM.append(testRRM)
                    pstRRm.append(testRRm)

    #print('corrected RR=',RRest,testRRm,testRRM) # to see the numbers for the evolution of Rt

    #print("anomaly resolved",i,testcim,new_cases,testciM) # the stats after anomaly resolution



    # visualization of the time evolution of R_t with confidence intervals
    plt.clf()
    x=[]
    for i in range(len(predR)):
        x.append(i)
    days=dates[3:]
    xd=days
    dstr=[]
    for xdd in xd:
        dstr.append(xdd.strftime("%Y-%m-%d"))
    # gg.writerow( (state,dstr,predR) )


    plt.title(state, fontsize=20, loc="left", fontdict={"family": "Fira Sans", "fontweight": "500"})
    plt.fill_between(xd,pstRRM, pstRRm,color='gray', alpha=0.3,label="99% Confidence Interval")

    plt.plot(xd,predR,'m-',lw=4,alpha=0.5,ms=8,label=r"Estimated $R_t$")
    plt.plot(xd,predR,'ro',ms=5,alpha=0.8)
    plt.plot(xd,pstRRM,'r-',alpha=0.8)
    plt.plot(xd,pstRRm,'r-',alpha=0.8)
    plt.plot( (xd[0],xd[-1]),(1,1),'k-',lw=4,alpha=0.4)
    plt.plot( (xd[-1],xd[-1]),(0.,max(predR)+1.),'k--')
    plt.text(xd[-1]+timedelta(days=1),1.1, 'today',fontsize=14)
    plt.xlim(xd[10],xd[-1]+timedelta(days=7))
    plt.ylim(0,max(predR))
    plt.ylabel(r"Estimated $R_t$",fontsize=14)
    plt.xlabel("Day",fontsize=20)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.legend()
    plt.gcf().set_size_inches(11, 8)
    plt.savefig(figs/f"Rt_Estimation_{state}.png", dpi=600, bbox_inches="tight")
    plt.close()
    #plt.show()


    # time series of new cases vs. real time prediction with anomalies
    plt.clf()
    fig, ax = plt.subplots()
    plt.title(state, fontsize=20, loc="left", fontdict={"family": "Fira Sans", "fontweight": "500"})
    print(state,len(xd),len(pred))
    plt.fill_between(xd,pstdM, pstdm,color='gray', alpha=0.3,label="99% Confidence Interval")
    plt.plot(xd,pred,'c-',lw=4,alpha=0.8,label="Expected Daily Cases")
    plt.plot(xd,pstdM,'k-',alpha=0.4)
    plt.plot(xd,pstdm,'k-',alpha=0.4)
    plt.plot(xd,NewCases,'bo',ms=7,alpha=0.3,label="Observed Cases")
    plt.plot(anomalyday,anomalypred,'o',c='red',ms=3,label="Anomalies")


    ## Now let's make predictions  into the future with and without control
    import numpy as np # This is here because otherwise I'm getting a type exception in the stats calls

    date = []
    date.append(xd[-1])
    tt=xd[-1]

    pop=1000000 #population size
    ndays=14 # number of days
    rintroductions=1. # number of infectious introduced in the population per day

    infperiod=5. # 5 days.
    gamma=1./infperiod
    m=0.02 # 2 percent case mortaliy
    DB=np.random.poisson(rintroductions,ndays) # vector of introductions, later to be from population movement matrix

    RR=np.ones(ndays)
    b=[]
    bNC=[]
    for i in range(len(RR)):
        RR[i]=predR[-1]
        b.append( np.exp( gamma*(RR[i]-1.) ) )
        bNC.append( np.exp( gamma*(RR[0]-1.) ) )
    print('Rt today=',RR[0])

    S=pop
    DeltaT=[]
    DeltaT.append(NewCases[-1]) # total cases, initialized with the first introduction, if any

    DeltaTNC=[]
    DeltaTNC.append(NewCases[-1]) # total cases, initialized with the first introduction, if any

    DeltaD=[]
    DeltaD.append(0.)

    DeltaR=[]
    DeltaR.append(0.)

    time=np.arange(ndays)

    D=0
    R=0
    I=100

    Susceptibles=[]
    Susceptibles.append(S)
    Infectious=[]
    Infectious.append(I)
    Dead=[]
    Dead.append(D)
    Recovered=[]
    Recovered.append(R)

    KP=0.05
    KI=0.15
    KD=0.0
    Dt=1.
    integral=0.
    u=0
    error0=RR[0]
    Rtarget=0.8
    upp=[]
    upp.append(pstdM[-1])
    lpp=[]
    lpp.append(pstdm[-1])

    uppNC=[]
    uppNC.append(pstdM[-1])
    lppNC=[]
    lppNC.append(pstdm[-1])


    for i in range(1,ndays):
        tt+=timedelta(days=1)
        date.append(tt)
        
        RR[i]=RR[i]*float(S)/float(pop) -u
        b[i]=np.exp( gamma*(RR[i]-1.) )
        #bNC[i]=np.exp( gamma*(RR[0]-1.) )
        error=RR[i]-Rtarget
        
        integral = integral + error * Dt
        derivative = (error - error0) / Dt
        
        u = KP*error + KI*integral + KD*derivative
        error0=error
        #print(RR[i],u,error)
        
        rateT=b[i]*DeltaT[i-1]
        Ncases=np.random.poisson(rateT,1)[0]
        
        rateNC=bNC[i]*DeltaTNC[i-1]
        NcasesNC=np.random.poisson(rateNC,1)[0]
        
        DeltaT.append(Ncases)
        DeltaTNC.append(NcasesNC)
        
        upper=poisson.ppf(0.99, rateT,1)
        lower=poisson.ppf(0.01, rateT,1)
        upp.append(upper)
        lpp.append(lower)

        upperNC=poisson.ppf(0.99, rateNC,1)
        lowerNC=poisson.ppf(0.01, rateNC,1)
        uppNC.append(upperNC)
        lppNC.append(lowerNC)
    
    

    plt.plot(date,DeltaT,color='green',marker='o',ms=6,alpha=.4,label=r'Projection w/ $R_{\rm t} \rightarrow 0.8$')
    plt.fill_between(date,upp, lpp,color='green', alpha=0.2)

    plt.plot(date,DeltaTNC,color='orange',marker='o',ms=6,alpha=.4,label=r'Projection w/ $R_{\rm t}=R_{\rm today}$')
    plt.fill_between(date,uppNC, lppNC,color='orange', alpha=0.2)



    plt.plot( (xd[-1],xd[-1]),(0.,max(DeltaT)+20.),'k--')
    plt.text(xd[-1]+timedelta(days=2),max(DeltaT), 'today',fontsize=14)

    plt.xlabel('Day',fontsize=20)
    plt.ylabel('New Cases',fontsize=20)
    plt.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.gcf().set_size_inches(11, 8)
    plt.gcf().set_size_inches(11, 8)
    plt.savefig(figs/f"Observed_Predicted_New_Cases_Control_Rtarget_{state}.png", dpi=600)
    plt.close()
    plt.close()
    #plt.show()
