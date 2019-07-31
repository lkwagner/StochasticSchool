import numpy as np
import scipy as sp
import scipy.linalg

npts=10
pos=np.random.randn(npts,3)
#pos holds npts points in 3 dimensions where x,y and z are chosen
#as normally distributed around zero

#want to find the distance form the origin for each point in a new array dist
dist=np.sqrt(np.sum(pos**2,axis=1))

#alternative solution
dist2=np.zeros(npts)
for i in np.arange(npts):
    dist2[i]=sp.linalg.norm(pos[i,:])

print(dist)
print(dist2)

