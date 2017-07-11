import numpy as np
from slaterwf import ExponentSlaterWF
from wavefunction import MultiplyWF,JastrowWF
from metropolis import metropolis_sample

def pair_distribution(wf):

  nelec=2
  ndim=3
  nconfig=1000
  nstep=100
  tau=0.2
  sample,acc = metropolis_sample(np.random.randn(nelec,ndim,nconfig),
                                 wf,tau=tau,nstep=nstep)
  dist=np.mean((sample[0]-sample[1])**2)**0.5
  print(dist)

if __name__=='__main__':
  wf=ExponentSlaterWF(1.0)
  pair_distribution(wf)

  wf=MultiplyWF(ExponentSlaterWF(1.0),JastrowWF(0.5))
  pair_distribution(wf)
