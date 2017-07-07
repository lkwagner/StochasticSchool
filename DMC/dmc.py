#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('../VMC/solutions')

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
      'weight':[],
      'eref':[],
      'r1':[],
      'r2':[],
      'r12':[]
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
    wtot=np.sum(weight)
    wavg=wtot/nconfig
    probability=np.cumsum(weight/wtot)
    randnums=np.random.random(nconfig)
    new_indices=np.searchsorted(probability,randnums)
    pos[:,:,new_indices]=pos
    weight[:]=wavg
#    print(pos)
    

    #Update the reference energy
    eref=eref-np.log(wavg)

    for i in range(nconfig):
      df['step'].append(istep)
      df['config'].append(i)
      df['elocal'].append(eloc[i])
      df['weight'].append(weight[i])
      df['eref'].append(eref)
      df['r1'].append(np.sum(pos[0,:,i]**2))
      df['r2'].append(np.sum(pos[1,:,i]**2))
      df['r12'].append(np.sum((pos[1,:,i]-pos[0,:,i])**2))
      
  return pd.DataFrame(df)


#####################################

if __name__ == '__main__':
  from slaterwf import ExponentSlaterWF
  from wavefunction import MultiplyWF, JastrowWF
  from hamiltonian import Hamiltonian
  nconfig=50
  df=simple_dmc(MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5)),
             Hamiltonian(),
             pos=np.random.randn(2,3,nconfig),
             tau=0.01,
             nstep=50000
             )
  df.to_csv("dmc.csv",index=False)

