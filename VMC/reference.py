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
  def __init__(self,Z=2):
    self.Z=Z
    pass
  def EN(self,x):
    r=np.sqrt(np.sum(x**2,axis=1))
    return np.sum(-self.Z/r,axis=0)
  def EE(self,x):
    ree=np.sqrt(np.sum((x[0,:,:]-x[1,:,:])**2,axis=0))
    return 1/ree
  def V(self,x):
    return self.EN(x)#+self.EE(x)
#------------------------------------

  
def MetropolisSample(x,wf,tau=0.01,nstep=1000):
  """
  Input variables:
    x: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    xnew: A 3D numpy array of configurations the same shape as x, distributed according to psi^2
    acceptance ratio: 
  """
  xnew=x.copy()
  xold=x.copy()
  wfold=wf.value(xold)
  acceptance=0.0
  nconf=x.shape[2]
  for s in range(nstep):
    xnew=xold+tau*np.random.standard_normal(xold.shape)
    wfnew=wf.value(xnew)
    acc=wfnew**2/wfold**2 + np.random.random_sample(nconf) > 1.0
    #print(acc)
    xold[:,:,acc]=xnew[:,:,acc]
    wfold[acc]=wfnew[acc]
    acceptance+=np.mean(acc)/nstep
    

  return xold,acceptance

#------------------------------------
def LocalEnergy(x,wf,H):
  return -0.5*np.sum(wf.laplacian(x),axis=0)+H.V(x)


if __name__=="__main__":
  nconfig=10000
  ndim=3
  nelec=2
  nstep=1000
  wf=wavefunction.ExponentSlaterWF(alpha=1.0)
  H=Hamiltonian(Z=1)
  xsample=np.random.randn(nelec,ndim,nconfig)
  xsample,acc=MetropolisSample(xsample,wf,tau=0.5,nstep=nstep)
  ke=-0.5*np.sum(wf.laplacian(xsample),axis=0)
  vion=H.EN(xsample)
  vee=H.EE(xsample)
  eloc=ke+vion+vee
  print("Cycle finished; acceptance",acc)
  refs=[1.0,-2.0,0.0,-1.0]
  for nm,quant,ref in zip(['kinetic','Electron-nucleus','Electron-electron','total'],[ke,vion,vee,eloc],refs):
    avg=np.mean(quant)
    err=np.std(quant)/np.sqrt(nconfig)
    print(nm,avg,"+/-",err, "reference",ref)


  

  
  

