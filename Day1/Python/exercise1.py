import numpy as np
import scipy as sp
import scipy.special

n=10
# fill in code here
# when x should be a random number between 1 and 2
# estimates should hold the first 10 partial sums of the taylor series of exp(x)


print("x=",x, "Exp(x)=",np.exp(x))
print(estimates)
print(estimates-np.exp(x))
