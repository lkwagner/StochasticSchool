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
  return dist

def compare_samealpha():
  wf=ExponentSlaterWF(2.0)
  slater_dist=pair_distribution(wf)

  wf=MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5))
  slaterjastrow_dist=pair_distribution(wf)

  print("Slater",slater_dist)
  print("Slater-Jastrow",slaterjastrow_dist)

def compare_optimal():
  wf=ExponentSlaterWF(1.7)
  slater_dist=pair_distribution(wf)

  wf=MultiplyWF(ExponentSlaterWF(1.9),JastrowWF(0.3))
  slaterjastrow_dist=pair_distribution(wf)

  print("Slater",slater_dist)
  print("Slater-Jastrow",slaterjastrow_dist)

if __name__=='__main__':
  print("Same alpha:")
  compare_samealpha()

  print()

  print("Optimal:")
  compare_optimal()
