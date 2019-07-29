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
    #generate points inside the unit square
    # fill this in
    for i in range(len(pts)):
        # check if points are inside the unit circle an if so record in acircle
        # fill this in
    #put the average number of points falling in the circle * 4 in estimate[j-1]
    #the error can be estimated using sp.stats.sem()
    #fill this in
    print (estimate[j-1], "+/-", error[j-1])
    
