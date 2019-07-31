import numpy as np; import scipy as sp
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt

nsamples=18 # number of sample sizes
estimate=np.zeros(nsamples)*1.0 # data arrays
error=np.zeros(nsamples)*1.0

for j in np.arange(nsamples)+1:
    npts=np.power(2,j) # double sampling points each time
    acircle=np.zeros(npts)
    pts=np.random.rand(npts,2)*2.0-1.0 # generate points inside unit square
    for i in range(len(pts)):
        if scipy.linalg.norm(pts[i]) < 1.0: # points inside the unit circle
            acircle[i]=1
    estimate[j-1]=np.mean(acircle)*4.0
    error[j-1]=sp.stats.sem(acircle)*4.0
    print (estimate[j-1], "+/-", error[j-1])
    
