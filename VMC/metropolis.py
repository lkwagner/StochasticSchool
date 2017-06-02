import numpy as np

def metropolis_sample(pos,wf,tau=0.01,nstep=1000):
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
  acceptance=0.0
  nconf=pos.shape[2]
  for istep in range(nstep):
    # propose a move
    gauss_move_old = np.random.randn(*posold.shape)
    posnew=posold+np.sqrt(tau)*gauss_move_old

    wfnew=wf.value(posnew)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    prob = wfnew**2/wfold**2 # for reversible moves

    # get indices of accepted moves
    acc_idx = (prob + np.random.random_sample(nconf) > 1.0)

    # update stale stored values for accepted configurations
    posold[:,:,acc_idx] = posnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

  return posold,acceptance

