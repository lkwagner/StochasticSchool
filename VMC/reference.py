#!/usr/bin/env python
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
  def pot_en(self,pos):
    """ electron-nuclear potential for configuration pos """
    r=np.sqrt(np.sum(pos**2,axis=1))
    return np.sum(-self.Z/r,axis=0)
  def pot_ee(self,pos):
    """ electron-electron potential for configuration pos """
    ree=np.sqrt(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0))
    return 1/ree
  def pot(self,pos):
    return self.pot_en(pos)+self.pot_ee(pos)

#####################################

def drift_vector(pos,wf,tau=None,scaled=False):
    """ return drift needed for electrons to importance sample psi^2 """
    vec = wf.gradient(pos) 
    dvec= vec # (grad_ln_psisq)/(2m) = grad_psi_over_psi
    if scaled: # rescale drift vector to limit its magnitude near psi=0
        if tau is None:
            raise RuntimeError('time step must be given to calculate scaled drift')
        vec_mag = np.sum(vec**2.,axis=1)
        # Umrigar, JCP 99, 2865 (1993).
        vscale  = (-1.+np.sqrt(1+2*vec_mag**2.*tau))/(vec_mag**2.*tau)
        dvec   *= vscale[:,np.newaxis,:]
    return dvec

def metropolis_sample(pos,wf,tau=0.01,nstep=1000,use_drift=False):
  """
  Input variables:
    pos: a 3D numpy array with indices (electron,[x,y,z],configuration ) 
    wf: a Wavefunction object with value(), gradient(), and laplacian()
  Returns: 
    posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
    acceptance ratio: 
  """

  # initialize
  posnew = pos.copy()
  posold = pos.copy()
  wfold  = wf.value(posold)
  if use_drift:
    driftold = drift_vector(posold,wf,tau=tau,scaled=True)
  acceptance=0.0
  nconf=pos.shape[2]
  for istep in range(nstep):

    # propose a move
    gauss_move_old = np.random.randn(*posold.shape)
    posnew=posold+tau*gauss_move_old
    if use_drift:
        posnew  += tau*driftold

    wfnew=wf.value(posnew)

    # calculate acceptance probability
    prob = wfnew**2/wfold**2
    if use_drift:
        driftnew = drift_vector(posnew,wf,tau=tau,scaled=True)
        gauss_move_new = (posold-posnew)/tau - driftnew
        #assert np.allclose( posold,posnew+tau*(gauss_move_new+driftnew) ) 
        gauss_old_sq = np.sum( np.sum(gauss_move_old**2.,axis=1) ,axis=0)
        gauss_new_sq = np.sum( np.sum(gauss_move_new**2.,axis=1), axis=0)
        ln_Tnew_over_Told = (gauss_old_sq-gauss_new_sq)/2.
        prob *= np.exp(ln_Tnew_over_Told)

    # get indices of accepted moves
    acc_idx = prob + np.random.random_sample(nconf) > 1.0

    # record
    posold[:,:,acc_idx]   = posnew[:,:,acc_idx]
    if use_drift:
        driftold[:,:,acc_idx] = driftnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

  return posold,acceptance

#####################################

def local_energy(pos,wf,ham):
  return -0.5*np.sum(wf.laplacian(pos),axis=0)+ham.pot(pos)

#####################################

def test_cusp(wf,ham):
  import matplotlib.pyplot as plt

  wf=wavefunction.JastrowWF(1.0)
  ham=Hamiltonian()

  # make sure jastrow has the right cusp
  assert(np.isclose(wf.Z,ham.Z))
  test_cusp(wf,ham)
  smallshift=1e-7
  path=np.zeros((2,3,1,100))+smallshift
  path[0,0,:,:]=np.linspace(-0.5,1,100)[np.newaxis,np.newaxis,:]
  path[1,0,:,:]=0.5+smallshift

  nuc_energy=ham.pot_en(path)
  elec_energy=ham.pot_ee(path)
  kin_energy=-0.5*np.sum(wf.laplacian(path),axis=0)
  tot_energy=kin_energy+nuc_energy+elec_energy

  fig,ax=plt.subplots(1,1)
  ax.plot(path[0,0,0,:],kin_energy[0],label='kinetic')
  ax.plot(path[0,0,0,:],nuc_energy[0],label='nuclear')
  ax.plot(path[0,0,0,:],elec_energy[0],label='interaction')
  ax.plot(path[0,0,0,:],tot_energy[0],label='total')
  ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),frameon=False)

  ax.set_ylim((-100,100))
  ax.set_xlabel('x-coordinate')
  ax.set_ylabel('Energy (Ha)')
  fig.set_size_inches(5,3)
  fig.tight_layout()
  fig.subplots_adjust(right=0.60)
  fig.savefig('cusp_test.pdf')

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
  possample,acc = metropolis_sample(possample,wf,tau=tau,nstep=nstep,use_drift=use_drift)

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

if __name__=="__main__":
  run_cusp_test()
  test_vmc(use_drift=False)
  test_vmc(use_drift=True)
