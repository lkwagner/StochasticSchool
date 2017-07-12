#!/usr/bin/env python
import numpy as np
import pandas as pd
from action import KineticAction, HarmonicPotentialAction

def action(paths,kaction,ext_pot):
  action = 0.0
  nslice,nptcl,ndim,nconf = paths.shape
  for jslice in range(nslice):
    action += kaction.kinetic_link_action(paths,jslice,(jslice+1)%nslice)
    action += ext_pot.potential_action(paths,jslice)
  return action

def metropolis_sample(paths,kaction,ext_pot,nstep=150,sigma=0.5):

  nslice,nptcl,ndim,nconf = paths.shape
  acceptance = 0.0
  for istep in range(nstep):
    # single slice move
    for islice in range(nslice):

      inext = (islice+1)%nslice
      iprev = (islice-1)%nslice

      # calculate action related to islice
      old_action = action(paths,kaction,ext_pot)

      # make a single slice move
      move = sigma*np.random.randn(nptcl,ndim,nconf)
      new_paths = paths.copy()
      new_paths[islice] += move

      # recalculate action related to islice
      new_action = action(new_paths,kaction,ext_pot)

      # accept or reject configurations
      action_change = new_action - old_action
      prob = np.exp(-action_change)
      acc_idx = prob>np.random.rand(nconf)
      paths[:,:,:,acc_idx] = new_paths[:,:,:,acc_idx]

      # record observables
      acceptance += np.mean(acc_idx)/nstep/nslice
  return acceptance,paths

def test_1d_sho(omegas=[5.,10.,15.,20.,25.],nconf=256,tau=0.05,nslice=20):

  nptcl  = 1
  ndim   = 1

  lam   = 0.5
  beta  = tau*nslice

  data = {'omega':[],'x2':[],'x2e':[]}
  for omega in omegas:
    paths   = np.random.randn(nslice,nptcl,ndim,nconf)
    kaction = KineticAction(tau,lam)
    ext_pot = HarmonicPotentialAction(tau,lam,omega)

    acc,new_paths = metropolis_sample(paths,kaction,ext_pot)
    print( "Cycle finished; acceptance = {acc:3.2f}.".format(acc=acc) )

    x2_val = np.mean(new_paths**2.)
    x2_err = np.std(new_paths**2.)/np.sqrt(nconf)
    data['omega'].append(omega)
    data['x2'].append(x2_val)
    data['x2e'].append(x2_err)

  df = pd.DataFrame(data)
  return df

if __name__ == '__main__':
  
  tau    = 0.1
  nslice = 10
  #tau    = 0.05
  #nslice = 20
  beta   = tau*nslice

  # get data
  dat_fname = '1d_sho.json'
  import os
  if not os.path.isfile(dat_fname):
    df = test_1d_sho(tau=tau,nslice=nslice)
    df.to_json(dat_fname)
  else:
    df = pd.read_json(dat_fname)
  # end if

  # compare to reference
  def analytic_x2(omega,beta):
    """ mean square deviation of 1D SHO with unit mass """
    return 1./(2*omega)*1./np.tanh(beta*omega/2.)

  xmin = df['omega'].min()
  xmax = df['omega'].max()
  finex = np.linspace(xmin,xmax,100)

  import matplotlib.pyplot as plt
  fig,ax = plt.subplots(1,1)
  ax.set_xlabel('omega',fontsize=16)
  ax.set_ylabel(r'<x$^2$>',fontsize=16)
  ax.set_xlim(xmin-1,xmax+1)

  ax.plot(finex,[analytic_x2(x,beta) for x in finex],lw=2,c='k',label='analytic')
  ax.errorbar(df['omega'],df['x2'],yerr=df['x2e'],fmt='x',mew=1,label='sampled')
  plt.show()


