#!/usr/bin/env python
import numpy as np
import pandas as pd
from action import primitive_action, exact_action

# ==== you will need to fill in metropolis_sample ====
def metropolis_sample(paths,tot_action,nstep=150,sigma=0.5):

  nslice,nptcl,ndim,nconf = paths.shape
  acceptance = 0.0
  for istep in range(nstep):
    # single slice moves
    for islice in range(nslice):

      # calculate action
      old_action = 0.0

      # make a single slice move
      new_paths = paths.copy()

      # recalculate action 
      new_action = 0.0

      # accept or reject configurations
      action_change = new_action - old_action
      prob = np.exp(-action_change)
      acc_idx = prob>np.random.rand(nconf)
      paths[:,:,:,acc_idx] = new_paths[:,:,:,acc_idx]

      # record acceptance rate
      acceptance += np.mean(acc_idx)/nstep/nslice
  return acceptance,paths
# ==== you will need to fill in metropolis_sample ====

def test_1d_sho(my_action,omegas=[5.,10.,15.,20.,25.],nconf=256,tau=0.05,nslice=20):

  nptcl  = 1
  ndim   = 1

  lam   = 0.5
  beta  = tau*nslice

  data = {'omega':[],'x2':[],'x2e':[]}
  for omega in omegas:
    paths   = np.random.randn(nslice,nptcl,ndim,nconf)
    tot_action = lambda x:my_action(x,omega,lam,tau)

    acc,new_paths = metropolis_sample(paths,tot_action)
    print( "Cycle finished; acceptance = {acc:3.2f}.".format(acc=acc) )

    x2_val = np.mean(new_paths**2.)
    x2_err = np.std(new_paths**2.)/np.sqrt(nconf)
    data['omega'].append(omega)
    data['x2'].append(x2_val)
    data['x2e'].append(x2_err)

  df = pd.DataFrame(data)
  return df

def compare_with_analytic(ax,df,beta):
  # compare to reference
  def analytic_x2(omega,beta):
    """ mean square deviation of 1D SHO with unit mass """
    return 1./(2*omega)*1./np.tanh(beta*omega/2.)

  xmin = df['omega'].min()
  xmax = df['omega'].max()
  finex = np.linspace(xmin,xmax,100)
  refy  = [analytic_x2(x,beta) for x in finex]

  ax.set_xlim(xmin-1,xmax+1)

  ref_line = ax.plot(finex,refy,lw=2,c='k',label='analytic')
  my_line = ax.errorbar(df['omega'],df['x2'],yerr=df['x2e'],fmt='x',mew=1,label='sampled')
  return ref_line,my_line

if __name__ == '__main__':
  
  tau    = 0.1
  nslice = 10
  #tau    = 0.05
  #nslice = 20
  beta   = tau*nslice

  # get data
  # ==========
  dat_fname = '1d_sho.csv'
  import os
  if not os.path.isfile(dat_fname):
    df = test_1d_sho(primitive_action,tau=tau,nslice=nslice)
    df.to_csv(dat_fname)
  else:
    df = pd.read_csv(dat_fname)
  # end if

  dat_fname1 = '1d_sho_exact.csv'
  import os
  if not os.path.isfile(dat_fname1):
    df1 = test_1d_sho(exact_action,tau=beta,nslice=1,nconf=4096)
    df1.to_csv(dat_fname1)
  else:
    df1 = pd.read_csv(dat_fname1)
  # end if

  # make plot
  # ==========
  import matplotlib.pyplot as plt
  fig,ax = plt.subplots(1,1)
  ax.set_xlabel('omega',fontsize=16)
  ax.set_ylabel(r'<x$^2$>',fontsize=16)

  ref_line,my_line   = compare_with_analytic(ax,df,beta)
  ref_line1,my_line1 = compare_with_analytic(ax,df1,beta)

  ax.legend(loc='upper right'
    ,handles=[ref_line[0],my_line,my_line1]
    ,labels=['analytic','sample primitve','sample exact']
  )
  plt.show()

