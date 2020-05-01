import csv
import sys
import scipy
import array
from matplotlib import *
from pylab import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt

def linreg(X, Y):
    """
        Summary
        Linear regression of y = ax + b
        Usage
        real, real, real = linreg(list, list)
        Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value
        """
    if len(X) != len(Y):  raise ValueError("unequal length")
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
    meanerror = residual = 0.0
    for x, y in zip(X, Y):
        meanerror = meanerror + (y - Sy/N)**2
        residual = residual + (y - a * x - b)**2
    RR = 1 - residual/meanerror
    ss = residual / (N-2)
    Var_a, Var_b = ss * N / det, ss * Sxx / det
    return a, b, RR, Var_a, Var_b

nation='india'
# here you can use nation='india', nation= 'united states', nation='canada' or any  other nation
#       or you can use a state such as nation='new york' or nation='illinois', which you could adapt to Indian states or districts... Girish Gupta


infperiod=4.5 # infectious period in days

g=open('virus.csv', 'r')
reader=csv.reader(g)

for row in reader:
    for i in range(len(row)):
        if (row[i]==nation):
            ii=i
g.close()

g=open('virus.csv', 'r')
reader=csv.reader(g)

day=[]
confirmed=[]
dead=[]
recovered=[]
i=0

for row in reader:
    i+=1
    str=row[ii].split("-")
    nn=len(str)
    print(str)
    if (str[0] and str[0]!=nation):
        print(i,row[0],str[0])
        day.append(float(i))
        confirmed.append(float(row[ii].split("-")[0]))
        print('confirmed',float(row[ii].split("-")[0]))
        if (nn>2 and row[ii].split("-")[2]):
            recovered.append(float(row[ii].split("-")[2]))
            print('recovered',float(row[ii].split("-")[2]))
        else:
            recovered.append(0.)
        
        
        if(nn>3 and row[ii].split("-")[3]):
            dead.append(float(row[ii].split("-")[3]))
            print('dead',float(row[ii].split("-")[3]))
        else:
            dead.append(0.)
g.close()

xx=[]
yy=[]
for i in range (len(confirmed)):
    if (confirmed[i]>0):
        xx.append(day[i])
        yy.append(np.log(confirmed[i]-recovered[i]-dead[i]))

plt.plot(xx,yy,'r-o',alpha=0.5,markersize=8)


print('days with cases=',len(yy))
R2=[]
growthrate=[]
egrowthrateM=[]
egrowthratem=[]
daysmeasured=[]
R=[]
RM=[]
Rm=[]

width=3
for ii in range(width,len(yy)):
    width=ii # a window with 4 days...
    iii=len(yy)-ii+width  #moving window? width = #, otherwise width =ii
    xxx=xx[-ii:iii]
    yyy=yy[-ii:iii]
    
    gradient, intercept, r_value, var_gr, var_it = linreg(xxx,yyy)
    #print( "R-squared", ii, r_value**2,gradient,2.*np.sqrt(var_gr),gradient*infperiod +1,xx[-ii])
    R2.append(r_value**2)
    growthrate.append(gradient)
    egrowthrateM.append(gradient+2.*np.sqrt(var_gr))
    egrowthratem.append(gradient-2.*np.sqrt(var_gr))
    
    daysmeasured.append(-(ii-3))
    R.append(gradient*4.5 +1.)
    RM.append( (gradient+2.*np.sqrt(var_gr))*infperiod +1.)
    Rm.append( (gradient-2.*np.sqrt(var_gr))*infperiod +1.)
    
# show models and best fit
    tt=xxx
    tt.sort()
    fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
    fity=intercept + fitx*gradient
    
    plt.plot(fitx,fity,'k-', linewidth=2, alpha=0.2)


plt.title(nation,fontsize=20)
plt.ylabel('infectious', fontsize=20)
plt.xlabel('time',fontsize=20)
plt.tight_layout()
savefig('Infectious.pdf')
plt.show()
plt.clf()

#extrapolate growth rate into the future
xx=daysmeasured[0:5]
yy=growthrate[0:5]

gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
#print(gradient,intercept,-intercept/gradient)
days2critical=int(-intercept/gradient)
print('days to critical',days2critical)

days2criticalm=int(-intercept/(gradient-2.*np.sqrt(var_gr)))
days2criticalM=int(-intercept/(gradient+2.*np.sqrt(var_gr)))

#print(days2critical,days2criticalM,days2criticalm)



tt=xx
tt.sort()
fitx=np.arange(float(tt[0]),float(tt[-1])+days2critical+1,0.1,dtype=float)
fity=intercept + fitx*gradient
    
plt.plot(fitx,fity,'g-', linewidth=2, alpha=0.9)
plt.plot(-intercept/gradient,0,'go',ms=10,alpha=1)


# plot cone of uncertainty
tt=xx
tt.sort()
fitx=np.arange(float(tt[0]),float(tt[-1])+days2criticalM,0.1,dtype=float)
fitym=intercept + fitx*(gradient-2.*np.sqrt(var_gr))

                   
tt=xx
tt.sort()
#fitx=np.arange(float(tt[0]),float(tt[-1])+days2critical,0.1,dtype=float)
fitx=np.arange(float(tt[0]),float(tt[-1])+days2criticalM,0.1,dtype=float)
fityM=intercept + fitx*(gradient+2.*np.sqrt(var_gr))


strd=np.str(days2critical)+' ['+ np.str(days2criticalm)+'-'+np.str(days2criticalM)+'] days'
plt.text(-10, 0.01, strd, fontsize=14,color='green',alpha=1.0)

plt.fill_between(fitx,fityM, fitym,color='green', alpha=0.3)
plt.fill_between(fitx,0., fitym,color='white', alpha=1.0)

# plot's labels and format
plt.title(nation,fontsize=20)
plt.plot(daysmeasured,growthrate,'ro-',alpha=0.6)
plt.fill_between(daysmeasured, egrowthratem, egrowthrateM,color='gray', alpha=0.3)
plt.text(daysmeasured[-1], 0.01, 'critical', fontsize=14,color='red',alpha=1.0)
#plt.plot( (days2critical,daysmeasured[-1]),(0.,0.),'k-',lw=3,alpha=0.4)
plt.plot( (daysmeasured[-1],days2criticalM),(0.,0.),'k-',lw=3,alpha=0.4)

plt.ylabel('growthrate', fontsize=20)
plt.xlabel('days from present',fontsize=20)
plt.tight_layout()
plt.ylim(-0.02,)
savefig('Growth_Rate_Extrapolation.pdf')
plt.show()
plt.clf()


plt.title(nation,fontsize=20)
plt.plot(daysmeasured,R2,'ko-')
plt.ylabel('R-squared', fontsize=20)
plt.xlabel('days from present',fontsize=20)
plt.tight_layout()
plt.show()


plt.clf()
plt.title(nation,fontsize=20)
plt.plot(daysmeasured,R,'ro-',alpha=0.6)
plt.fill_between(daysmeasured, Rm, RM,color='gray', alpha=0.3)
plt.plot( (0,daysmeasured[-1]),(1.,1.),'k-',lw=3,alpha=0.4)
plt.text(daysmeasured[-1], 1.03, 'critical', fontsize=14,color='red',alpha=1.0)
plt.ylabel('Reproductive Rate', fontsize=20)
plt.xlabel('days from present',fontsize=20)
plt.tight_layout()
savefig('Reproducive_Rate_vs_Critical.pdf')
plt.show()
