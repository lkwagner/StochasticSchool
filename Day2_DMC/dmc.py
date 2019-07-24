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


def acceptance(posold,posnew,driftold,driftnew,tau,wf):
    """Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after  move (nelec,ndim,nconf)
      driftnew: drift vector at posnew 
      tau: time step
      wf: wave function object
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """
    gfratio=np.exp(-np.sum( (posold-posnew-driftnew)**2/(2*tau),axis=(0,1)) 
                   +np.sum( (posnew-posold-driftold)**2/(2*tau),axis=(0,1)) )
    ratio=wf.value(posnew)**2/wf.value(posold)**2
    return ratio*gfratio

def simple_dmc(wf,ham,tau,pos,nstep=1000):
  """
  Inputs:
  
  Outputs:
  A Pandas dataframe with each 

  """
  df={'step':[],
      'elocal':[],
      'weight':[],
      'weightvar':[],
      'elocalvar':[],
      'eref':[],'tau':[]
      }
  nconfig=pos.shape[2]
  pos,acc=metropolis_sample(pos,wf,tau=0.5)
  weight=np.ones(nconfig)
  ke,pot,eloc=ke_pot_tot_energies(pos,wf,ham)
  eref=np.mean(eloc)
  
  for istep in range(nstep):
    #Drift+diffusion
    driftold=tau*wf.gradient(pos)
    ke,pot,elocold=ke_pot_tot_energies(pos,wf,ham)
    
    posnew=pos+np.sqrt(tau)*np.random.randn(*pos.shape)+driftold
    driftnew=tau*wf.gradient(posnew)
    acc=acceptance(pos,posnew,driftold,driftnew,tau,wf)
    imove=acc>np.random.random(nconfig)
    pos[:,:,imove]=posnew[:,:,imove]
    acc_ratio=np.sum(imove)/nconfig
    
    #Change weight
    ke,pot,eloc=ke_pot_tot_energies(pos,wf,ham)
    weight*=np.exp(-0.5*tau*(eloc+elocold-2*eref))
    
    #Branch
    wtot=np.sum(weight)
    wavg=wtot/nconfig
    probability=np.cumsum(weight/wtot)
    randnums=np.random.random(nconfig)
    new_indices=np.searchsorted(probability,randnums)
    posnew=pos[:,:,new_indices]
    pos=posnew.copy()
    

    #Update the reference energy
    eref=eref-np.log(wavg)

    print("average energy",np.mean(eloc*weight/wavg),"eref",eref,"acceptance",acc_ratio)
    df['step'].append(istep)
    df['elocal'].append(np.mean(eloc))
    df['weight'].append(np.mean(weight))
    df['elocalvar'].append(np.std(eloc))
    df['weightvar'].append(np.std(weight))
    df['eref'].append(eref)
    df['tau'].append(tau)
    weight.fill(wavg)
    
      
  return pd.DataFrame(df)


#####################################

if __name__ == '__main__':
  from slaterwf import ExponentSlaterWF
  from wavefunction import MultiplyWF, JastrowWF
  from hamiltonian import Hamiltonian
  nconfig=50000
  dfs=[]
  for tau in [.01,.005,.0025]:
    dfs.append(simple_dmc(MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5)),
             Hamiltonian(),
             pos=np.random.randn(2,3,nconfig),
             tau=tau,
             nstep=10000
             ))
  df=pd.concat(dfs)
  df.to_csv("dmc.csv",index=False)

