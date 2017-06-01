#!/usr/bin/env python
import numpy as np
import sys
sys.path.insert(0,'../VMC')
sys.path.insert(0,'../MonteCarlo')
from analyze_trace import error
from wavefunction import ExponentSlaterWF, JastrowWF, MultiplyWF
from reference import Hamiltonian,local_energy,metropolis_sample,\
  drift_vector,drift_prob

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

def avg_by_weight(val_vec,weights):
    """ average value vector by weights
     both val_vec and weights should have shape (nconfig,) 
     return: one float, the weighted average of val_vec """
    return (val_vec*weights).sum()/weights.sum()

#####################################

def weight_dmc(pos,wf,ham,weights=None,weight_desire=None,tau=0.1,nstep=1000,use_drift=True,kappa=1.0):
  """ project pos guided by wf using exp(-tau*ham) for nstep
  Inputs:
    wf: wavefunction object with value(), gradient(), and laplacian() methods
      to calculate psi, grad_psi_over_psi, and lap_psi_over_psi
    ham: hamiltonian object with pot() to evaluate the potential energy of a 
      configuration
    pos: starting configurations having shape (nelec,ndim,nconfig)
      hopefully sampled from the wavefunction using VMC
    weights: weights of configurations having shape (nconfig,)
      hopefully from a previous DMC run, otherwise initialize to np.ones
    tau: timestep
    nstep: number of steps
    use_drift: use importance sampling transformation
    kappa: correlation length in imaginary time, used in population control
  Returns:
    posnew:  shape (nelec,ndim,nconf) configurations after projection
    weights: shape (nconf,) accumulated weights of each configuration
    traces:  shape (nstep,nobs) traces of observables during projection
  """
  # initialize
  posnew = pos.copy()
  posold = pos.copy()
  wfold  = wf.value(posold)
  if use_drift:
    driftold = drift_vector(posold,wf,tau=tau,scaled=True)
  nconf=pos.shape[2]
  if weights is None:
      weights = np.ones(nconf)
      weight_desire = weights.sum()
  acceptance=0.0

  obs_idx= {'eloc':0,'etrial':1,'weight':2}
  traces = np.zeros([nstep,len(obs_idx)])

  elocold    = local_energy(posold,wf,ham)
  etrial     = avg_by_weight(elocold,weights)

  for istep in range(nstep):

    # drift-diffusion move
    gauss_move_old = np.random.randn(*posold.shape)
    posnew=posold+np.sqrt(tau)*gauss_move_old
    if use_drift:
        posnew  += tau*driftold

    wfnew=wf.value(posnew)
    # calculate energies
    kenew,potnew,elocnew = ke_pot_tot_energies(posnew,wf,ham)
    # grow weights
    weights *= np.exp(-tau*(elocnew-etrial))

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    #  DMC uses rejection to enforce continuity of psi^2, not necessary
    prob = wfnew**2/wfold**2 # for reversible moves
    if use_drift: # multiply ratio of probabilities of backward/forward moves
        driftnew = drift_vector(posnew,wf,tau=tau,scaled=True)
        prob *= drift_prob(posold,posnew,gauss_move_old,driftnew,tau)

    # get indices of accepted moves
    acc_idx = prob + np.random.random_sample(nconf) > 1.0

    # update stale stored values for accepted configurations
    posold[:,:,acc_idx]   = posnew[:,:,acc_idx]
    if use_drift:
        driftold[:,:,acc_idx] = driftnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    elocold[acc_idx] = elocnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

    eloc_mean = avg_by_weight(elocnew,weights)
    observables = {'eloc':eloc_mean,'etrial':etrial,'weight':weights.sum()}
    for name in observables.keys():
        idx = obs_idx[name]
        traces[istep,idx] = observables[name]

    # population control
    etrial = elocold.mean() + np.log(weight_desire/weights.sum())/kappa
  
  return posnew,weights,acceptance,traces

#####################################

def test_dmc(
    dmc_method,
    trial_wf= JastrowWF(100.,Z=2.0),
    nconf = 10000,  # number of walkers
    proj_time = 5., # projection time in a.u.
    tau_list= [0.16,0.08,0.04]
    ):

    # hard-code He
    nelec   = 2
    ndim    = 3
    Zcharge = 2

    # initialize hamiltonian
    he_ham = Hamiltonian(Z=Zcharge)

    # use VMC to start sample configurations from trial_wf
    pos0 = np.random.randn(nelec,ndim,nconf) 
    pos_vmc,acc_vmc = metropolis_sample(pos0,trial_wf,tau=0.5,nstep=100)

    # run DMC to project the VMC wavefunction to ground state
    data = []
    pos_last     = pos_vmc.copy()
    weights_last = np.ones(nconf)
    for tau in tau_list:
        nstep = int(round(proj_time/tau))
        half  = int(round(nstep/2)) # assume equilibration at half time
        pos_dmc,weights_dmc,acc_dmc,traces = dmc_method(
          pos_last,trial_wf,he_ham,
          weights=weights_last,weight_desire=nconf,
          tau=tau,nstep=nstep
        )
        print( "tau={tau:6.3f}: {acc:d}% {emean:10.6f} +- {err:10.6f}".format(
            tau=tau,acc=int(round(acc_dmc*100)),emean=np.mean(traces[half:,0]),
            err=error(traces[half:,0]) ) )
        data.append(traces)
        # restart with current walkers
        weights_last = weights_dmc
        pos_last = pos_dmc

    return np.concatenate(data,axis=0)

#####################################

if __name__ == '__main__':
    traces = test_dmc(weight_dmc)
    with open('traces.dat','w') as f:
      f.write('# eloc etrial weight\n')
      np.savetxt(f,traces) 
