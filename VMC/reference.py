#This is not the most efficient (in memory or computer time) implementation. 
#It is supposed to be relatively straightforward to 
#understand and flexible enough to play with.
#We will have to decide how much we want to give them 
#(the structure, library functions, etc), 
#and how much we want to have them develop.
import numpy as np
import wavefunction

#####################################

class Hamiltonian:
  def __init__(self,Z=2):
    self.Z=Z
    pass
  def EN(self,pos):
    r=np.sqrt(np.sum(pos**2,axis=1))
    return np.sum(-self.Z/r,axis=0)
  def EE(self,pos):
    ree=np.sqrt(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0))
    return 1/ree
  def V(self,pos):
    return self.EN(pos)#+self.EE(pos)

#####################################
  
def metropolis_sample(pos,wf,tau=0.01,nstep=1000):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """
  posnew=pos.copy()
  posold=pos.copy()
  wfold=wf.value(posold)
  acceptance=0.0
  nconf=pos.shape[2]
  for s in range(nstep):
    posnew=posold+tau*np.random.standard_normal(posold.shape)
    wfnew=wf.value(posnew)
    acc=wfnew**2/wfold**2 + np.random.random_sample(nconf) > 1.0
    #print(acc)
    posold[:,:,acc]=posnew[:,:,acc]
    wfold[acc]=wfnew[acc]
    acceptance+=np.mean(acc)/nstep

  return posold,acceptance

#####################################

def local_energy(pos,wf,H):
  return -0.5*np.sum(wf.laplacian(pos),axis=0)+H.V(pos)

#####################################

#def test_cusp(wf,H):

#####################################

def test_vmc(
    nconfig=10000,
    ndim=3,
    nelec=2,
    nstep=1000,
    wf=wavefunction.ExponentSlaterWF(alpha=1.0),
    H=Hamiltonian(Z=1)):
  ''' Calculate VMC energies and compare to reference values.'''

  possample=np.random.randn(nelec,ndim,nconfig)
  possample,acc=metropolis_sample(possample,wf,tau=0.5,nstep=nstep)
  ke=-0.5*np.sum(wf.laplacian(possample),axis=0)
  vion=H.EN(possample)
  vee=H.EE(possample)
  eloc=ke+vion+vee
  print("Cycle finished; acceptance",acc)
  refs=[1.0,-2.0,0.0,-1.0]
  for nm,quant,ref in zip(['kinetic','Electron-nucleus','Electron-electron','total'],[ke,vion,vee,eloc],refs):
    avg=np.mean(quant)
    err=np.std(quant)/np.sqrt(nconfig)
    print(nm,avg,"+/-",err, "reference",ref)

if __name__=="__main__":

  test_vmc()
