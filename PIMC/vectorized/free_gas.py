import numpy as np
import pandas as pd
from action import primitive_action
from metropolis import metropolis_sample
from observe import draw_beads_3d,thermodynamic_kinetic

def generate_free_gas_paths(
   tau    = 0.3
  ,nslice = 4
  ,nptcl  = 3
  ,ndim   = 3 
  ,nconf  = 16
  ,nstep  = 300
  ,lam    = 0.5
  ,visualize = True
  ):

  omega = 0.0 # no potential
  beta = nslice*tau # inverse temperature

  # initialize paths randomly
  paths  = np.random.randn(nslice,nptcl,ndim,nconf)

  # initialize free action
  tot_action = lambda x:primitive_action(x,omega,lam,tau)

  # use free action to sample paths
  acc,new_paths = metropolis_sample(paths,tot_action,nstep=300)

  return beta,new_paths

def n_free_slices(nslice,nconf=16,lam=0.5,visualize=True):
  # sample free density matrix
  beta,paths = generate_free_gas_paths(nslice=nslice,lam=lam,nconf=nconf)

  # calculate kinetic energy
  nslice,nptcl,ndim,nconf = paths.shape
  kall    = thermodynamic_kinetic(paths,lam,float(beta)/nslice)
  ke_mean = kall.mean()/nptcl/ndim
  ke_err  = kall.std()/nptcl/ndim/np.sqrt(nconf)
  print( 'kinetic = %8.6f +- %8.6f per particle per dimension' % (ke_mean,ke_err) )
  print( ' expect   %8.6f per degree of freedom at beta = %4.2f' % (0.5/beta,beta) )

  if visualize:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # enable 3D

    iconf = 0 # configuration to show
    beads = paths[:,:,:,iconf]

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1,projection='3d',aspect=1)
    ax.set_xlabel('x',fontsize=16)
    ax.set_ylabel('y',fontsize=16)
    ax.set_zlabel('z',fontsize=16)

    draw_beads_3d(ax,beads)

    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    fig.tight_layout()
    plt.show()

  return beta,ke_mean,ke_err

def test_free_kinetic(nslices=[2,4,8,16],nconf=16):
  data = {'beta':[],'ke_mean':[],'ke_error':[]}
  for nslice in nslices:
    beta,ke_mean,ke_err = n_free_slices(nslice=nslice,nconf=nconf,visualize=False)
    data['beta'].append(beta)
    data['ke_mean'].append(ke_mean)
    data['ke_error'].append(ke_err)
  # end for nslice
  df = pd.DataFrame(data)
  return df

if __name__ == '__main__':
  
  n_free_slices(5)
  ready_for_next_step = False
  if not ready_for_next_step:
    assert 1==0

  nconf = 256
  import os
  dat_fname = 'gas3d_free.csv'
  if not os.path.isfile(dat_fname):
    df = test_free_kinetic(nconf=nconf)
    df.to_csv(dat_fname)
  else:
    df = pd.read_csv(dat_fname)
  df.sort_values('beta',inplace=True)

  ref = 0.5/df['beta'].values
  myy = df['ke_mean'].values
  myye= df['ke_error'].values
   
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax  = fig.add_subplot(1,1,1,aspect=1)

  ax.errorbar(ref,myy,yerr=myye,fmt='o')
  ax.plot([ref[0],ref[-1]],[myy[0],myy[-1]],c='k',lw=2,ls='--')

  ax.set_xlabel('PIMC kinetic energy (a.u.)',fontsize=16)
  ax.set_ylabel('Virial kinetic energy (a.u.)',fontsize=16)
  ax.set_xlim(-0.2,1.2)
  ax.set_ylim(-0.2,1.2)
  plt.show()

