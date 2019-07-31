import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_csv("cn-energies.csv")

massa=12.011
massb=14.007
mu=massa*massb/(massa+massb)
#need to convert mu to proper units
mu=mu/9.10938291E-28/6.0221413E23

# conversion factors
evtocm1=8065.54429
bohrtoang=0.529177
hatoev=27.211396132



def morse(x,a,b,c,d):
    return a*(np.exp(-2*b*(x-d))-2*np.exp(-b*(x-d)))+c

def omega(aa,bb,m):
    return (bb*np.sqrt(2*aa/m)-bb*bb/m)

# need to use good guesses for the parameters
p0=(0.445,1.1,-15.1,2.2)

popt, pcov = curve_fit(morse,df.r,df.en,p0,df.err)

a=popt[0]
b=popt[1]
c=popt[2]
d=popt[3]


#plot the fit
#plt.errorbar(df.r,df.en,yerr=df.err)
#plt.plot(rvals,morse(df.r,*popt), 'g--', label='fit-with-bounds')
#plt.show()

print("equilibrium distance = ", d*bohrtoang, "angstroms")
print("vibrational frequency = ", omega(a,b,mu)*hatoev*evtocm1, "cm^-1")
