#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('../VMC')

from metropolis import metropolis_sample
import pandas as pd

#####################################

def ke_pot_tot_energies(pos,wf,ham):
    """ calculate kinetic, potential, and local energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
    Return:
      ke: kinetic energy
      pot: potential energy
      eloc: local energy
    """
    ke   = -0.5*np.sum(wf.laplacian(pos),axis=0)
    pot  = ham.pot(pos)
    eloc = ke+pot
    return ke,pot,eloc

#####################################

def simple_dmc(wf,ham,tau,pos,nstep=1000):
  """
  Inputs:
  
  Outputs:
  A Pandas dataframe with each 

  """
  df={'step':[],
      'config':[],
      'elocal':[],
      'weight':[]
      }
  nconfig=pos.shape[2]
  pos,acc=metropolis_sample(pos,wf,tau=0.5)
  weight=np.ones(nconfig)
  ke,pot,eloc=ke_pot_tot_energies(pos,wf,ham)
  eref=np.mean(eloc)
  
  for istep in range(nstep):
    #Drift+diffusion
    pos+=np.sqrt(tau)*np.random.randn(*pos.shape)+tau*wf.gradient(pos)
    #Change weight
    ke,pot,eloc=ke_pot_tot_energies(pos,wf,ham)
    weight*=np.exp(-tau*(eloc-eref))
    
    #Branch

    #Update the reference energy
    

    for i in range(nconfig):
      df['step'].append(istep)
      df['config'].append(i)
      df['elocal'].append(eloc[i])
      df['weight'].append(weight[i])
      
  return pd.DataFrame(df)


#####################################

if __name__ == '__main__':
  from slaterwf import ExponentSlaterWF
  from wavefunction import MultiplyWF, JastrowWF
  from hamiltonian import Hamiltonian
  nconfig=1000
  df=simple_dmc(MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5)),
             Hamiltonian(),
             pos=np.random.randn(2,3,nconfig),
             tau=0.01,
             nstep=10000
             )
  df.to_csv("dmc.csv",index=False)

