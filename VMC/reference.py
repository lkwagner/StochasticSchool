#This is not the most efficient (in memory or computer time) implementation. 
#It is supposed to be relatively straightforward to 
#understand and flexible enough to play with.
#We will have to decide how much we want to give them 
#(the structure, library functions, etc), 
#and how much we want to have them develop.
import numpy as np
import wavefunction

#------------------------------------

class Hamiltonian:
  def __init__(self):
    pass
  def V(self):
    pass
#------------------------------------

  
def MetropolisSample(x,wf):
  """
  Input variables:
    x: a 3D numpy array with indices (configuration,electron,[x,y,z] ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    xnew: A 3D numpy array of configurations the same shape as x, distributed according to psi^2
  """
  pass

#------------------------------------
def LocalEnergy(x,wf,H):
  return -0.5*wf.laplacian(x)+H.V(x)


#How this might be used

if __name__=="__main__":
  nconfig=100
  ndim=3
  nelec=2
  wf=SlaterWf(alpha=1.0)
  H=Hamiltonian()
  #start all electrons at zero
  x0=np.zeros((nconfig,nelec,ndim)) 
  xsample=MetropolisSample(x,wf)
  eloc=LocalEnergy(xsample,wf,H)
  eavg=np.mean(eloc)
  eerr=np.std(eloc)/np.sqrt(nconfig)
  print("eavg",eavg,"+/-",eerr)
  

  
  

