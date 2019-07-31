import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

timestep=0.01
diffusionConstant=0.5
popsize=128
numsteps=1000

energies=np.zeros(numsteps)
avgpos=np.zeros(numsteps)
spread=np.zeros(numsteps)


a=0.1
b=-1
c=-0.1
d=0.05

def val(x):
    fun =  a*x + b*x*x + c*x*x*x + d*x*x*x*x
    return fun

def deriv(x):
    der = a + 2*b*x + 3*c*x*x + 4*d*x*x*x
    return der

def noop(x):
    return x

def spr(loc):
    return np.std(loc)

def popval(loc, fun):
    values = fun(loc)
    return np.mean(values)

def updateLoc(loc, tau, diffconst, df):
    newloc = loc - tau*df(loc) + np.sqrt(tau)*diffconst*np.random.randn(popsize)
    return newloc;


# start walkers randomly distributed between 2 and 3
loc = np.random.rand(popsize)
loc += 2

print(0,popval(loc,val),popval(loc,noop))

for x in range(numsteps):
    loc = updateLoc(loc, timestep, diffusionConstant, deriv)
    energies[x] = popval(loc,val)
    spread[x] = spr(loc)
    print(x+1,energies[x],popval(loc,noop))

# Use Seaborn to histogram of walker positions at end    
#sns.set_style("white")
#sns.distplot(loc)
#plt.show()

# Use Pandas to save timeseries data to a csv file
#df = pd.DataFrame({'energies': energies,
#                   'spread': spread})
#
#df.to_csv("first_timeseries.csv",index=True)

# Use matplotlib to plot timeseries data
plt.plot(energies)
plt.xlim([0,1.2/timestep]) 
plt.show()
