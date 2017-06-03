import numpy as np

def drift_vector(pos,wf,tau,scaled=False):
    """ calculate the drift vector for importance sampling
    Input:
       pos: 3D numpy array of electron positions (nelec,ndim,nconf)
       wf:  wavefunction to be sampled (i.e. psi)
    Return: 
       dvec: drift vector needed for electrons to importance sample psi^2
    """

    # drift vector = (\nabla^2\ln\psi^2)/(2m) = grad_psi_over_psi
    dvec= wf.gradient(pos) 
    if scaled: # rescale drift vector to limit its magnitude near psi=0
        vec_sq = np.sum(dvec**2.,axis=1)
        # Umrigar, JCP 99, 2865 (1993).
        vscale  = (-1.+np.sqrt(1+2*vec_sq*tau))/(vec_sq*tau)
        dvec   *= vscale[:,np.newaxis,:]
    return dvec

def drift_prob(posold,posnew,gauss_move_old,driftnew,tau):
    """ return the ratio of forward and backfward move probabilities for rejection algorith,
    Input:
      posold: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after  move (nelec,ndim,nconf)
      gauss_move_old: randn() numbers for forward move
      driftnew: drift vector at posnew 
      tau: time step
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """

    # randn numbers needed for backward move
    gauss_move_new = (posold-posnew-tau*driftnew)/np.sqrt(tau)
    # assume the following drift-diffusion move
    #assert np.allclose( posold,posnew+np.sqrt(tau)*gauss_move_new+tau*driftnew ) 

    # calculate move probabilities
    gauss_old_sq = np.sum( np.sum(gauss_move_old**2.,axis=1) ,axis=0)
    gauss_new_sq = np.sum( np.sum(gauss_move_new**2.,axis=1), axis=0)
    forward_green  = np.exp(-gauss_old_sq/2.)
    backward_green = np.exp(-gauss_new_sq/2.)

    ratio = backward_green/forward_green
    return ratio

#####################################
def metropolis_sample_biased(pos,wf,tau=0.01,nstep=1000):
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
  driftold = drift_vector(posold,wf,tau=tau,scaled=True)
  acceptance=0.0
  nconf=pos.shape[2]
  for istep in range(nstep):

    # propose a move
    gauss_move_old = np.random.randn(*posold.shape)
    posnew=posold+np.sqrt(tau)*gauss_move_old
    posnew += tau*driftold

    wfnew=wf.value(posnew)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    #  VMC uses rejection to sample psi_sq by maintaining detailed balance
    prob = wfnew**2/wfold**2 # for reversible moves
    driftnew = drift_vector(posnew,wf,tau=tau,scaled=True)
    prob *= drift_prob(posold,posnew,gauss_move_old,driftnew,tau)

    # get indices of accepted moves
    acc_idx = (prob + np.random.random_sample(nconf) > 1.0)

    # update stale stored values for accepted configurations
    posold[:,:,acc_idx] = posnew[:,:,acc_idx]
    driftold[:,:,acc_idx] = driftnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

  return posold,acceptance

