#!/usr/bin/env python
#This is not the most efficient (in memory or computer time) implementation. 
#It is supposed to be relatively straightforward to 
#understand and flexible enough to play with.
#We will have to decide how much we want to give them 
#(the structure, library functions, etc), 
#and how much we want to have them develop.
import numpy as np
import wavefunction
from hamiltonian import Hamiltonian
from metropolis import metropolis_sample


#####################################


def local_energy(pos,wf,ham):
  return -0.5*np.sum(wf.laplacian(pos),axis=0)+ham.pot(pos)

#####################################

def test_vmc(
    nconfig=1000, ndim=3,
    nelec=2,
    nstep=100,
    tau=0.5,
    wf=wavefunction.ExponentSlaterWF(alpha=1.0),
    ham=Hamiltonian(Z=1),
    use_drift=False
    ):
  ''' Calculate VMC energies and compare to reference values.'''

  print( 'VMC test: 2 non-interacting electrons around a fixed proton' )

  # initialize electrons randomly
  possample     = np.random.randn(nelec,ndim,nconfig)
  # sample exact wave function
  if use_drift:
    possample,acc = metropolis_sample(possample,wf,tau=tau,nstep=nstep)
  else:
    possample,acc = metropolis_sample_biased(possample,wf,tau=tau,nstep=nstep)

  # calculate kinetic energy
  ke   = -0.5*np.sum(wf.laplacian(possample),axis=0)
  # calculate potential energy
  vion = ham.pot_en(possample)
  eloc = ke+vion

  # report
  print( "Cycle finished; acceptance = {acc:3.2f}.".format(acc=acc) )
  for nm,quant,ref in zip(['kinetic','Electron-nucleus','total']
                         ,[ ke,       vion,              eloc]
                         ,[ 1.0,      -2.0,              -1.0]):
    avg=np.mean(quant)
    err=np.std(quant)/np.sqrt(nconfig)
    print( "{name:20s} = {avg:10.6f} +- {err:8.6f}; reference = {ref:5.2f}".format(
      name=nm, avg=avg, err=err, ref=ref) )

#####################################

def test_hellium(
    nconfig=1000, ndim=3,
    nelec=2,
    nstep=100,
    tau=0.5,
    wf=wavefunction.ExponentSlaterWF(alpha=1.0),
    ham=Hamiltonian(Z=2),
    use_drift=False
    ):
  ''' Calculate VMC energies and compare to reference values.'''

  print( 'VMC test: Hellium ground state' )

  # initialize electrons randomly
  possample     = np.random.randn(nelec,ndim,nconfig)
  # sample trial wave function
  possample,acc = metropolis_sample(possample,wf,tau=tau,nstep=nstep)

  # calculate kinetic energy
  ke   = -0.5*np.sum(wf.laplacian(possample),axis=0)
  # calculate potential energy
  pot  = ham.pot(possample)
  eloc = ke+pot

  # report
  print( "Cycle finished; acceptance = {acc:3.2f}.".format(acc=acc) )
  for nm,quant,ref in zip(['kinetic','potential','total']
                         ,[ ke,       pot,        eloc]
                         ,[ 2.848, -5.696,      -2.848]):
    avg=np.mean(quant)
    err=np.std(quant)/np.sqrt(nconfig)
    print( "{name:20s} = {avg:10.6f} +- {err:8.6f}; reference = {ref:5.2f}".format(
      name=nm, avg=avg, err=err, ref=ref) )
#####################################

if __name__=="__main__":
  test_cusp()
  test_vmc(use_drift=False)
  test_vmc(use_drift=True)
  test_hellium(tau=0.4,use_drift=True,wf=wavefunction.JastrowWF(1e9))
