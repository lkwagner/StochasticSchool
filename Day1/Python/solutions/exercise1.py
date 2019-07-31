import numpy as np
import scipy as sp
import scipy.special
n=10
estimates=np.arange(n,dtype='float64')
x=np.random.random()+1
for i in range(n):
    d=np.arange(i)
    c=1.0/sp.special.factorial(d)
    d=np.power(x,d)
    estimates[i]=np.sum(np.multiply(c,d))

print("x=",x, "Exp(x)=",np.exp(x))
print(estimates)
print(estimates-np.exp(x))
