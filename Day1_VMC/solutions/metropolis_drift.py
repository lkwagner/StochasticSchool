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

def drift_prob(poscur,posnew,gauss_move_cur,driftnew,tau):
    """ return the ratio of forward and backfward move probabilities for rejection algorith,
    Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after  move (nelec,ndim,nconf)
      gauss_move_cur: randn() numbers for forward move
      driftnew: drift vector at posnew 
      tau: time step
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """

    # randn numbers needed for backward move
    gauss_move_new = (poscur-posnew-tau*driftnew)/np.sqrt(tau)
    # assume the following drift-diffusion move
    #assert np.allclose( poscur,posnew+np.sqrt(tau)*gauss_move_new+tau*driftnew ) 

    # calculate move probabilities
    gauss_cur_sq = np.sum( np.sum(gauss_move_cur**2.,axis=1) ,axis=0)
    gauss_new_sq = np.sum( np.sum(gauss_move_new**2.,axis=1), axis=0)
    forward_green  = np.exp(-gauss_cur_sq/2.)
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
  poscur = pos.copy()
  wfcur  = wf.value(poscur)
  driftcur = drift_vector(poscur,wf,tau=tau,scaled=True)
  acceptance=0.0
  nconf=pos.shape[2]
  for istep in range(nstep):

    # propose a move
    gauss_move_cur = np.random.randn(*poscur.shape)
    posnew=poscur+np.sqrt(tau)*gauss_move_cur
    posnew += tau*driftcur

    wfnew=wf.value(posnew)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    #  VMC uses rejection to sample psi_sq by maintaining detailed balance
    prob = wfnew**2/wfcur**2 # for reversible moves
    driftnew = drift_vector(posnew,wf,tau=tau,scaled=True)
    prob *= drift_prob(poscur,posnew,gauss_move_cur,driftnew,tau)

    # get indices of accepted moves
    acc_idx = (prob + np.random.random_sample(nconf) > 1.0)

    # update stale stored values for accepted configurations
    poscur[:,:,acc_idx] = posnew[:,:,acc_idx]
    driftcur[:,:,acc_idx] = driftnew[:,:,acc_idx]
    wfcur[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

  return poscur,acceptance

def test_metropolis(
      nconfig=1000,
      ndim=3,
      nelec=2,
      nstep=100,
      tau=0.5
    ):
  from slaterwf import ExponentSlaterWF
  from hamiltonian import Hamiltonian
  
  wf=ExponentSlaterWF(alpha=1.0)
  ham=Hamiltonian(Z=1)
  
  possample     = np.random.randn(nelec,ndim,nconfig)
  possample,acc = metropolis_sample_biased(possample,wf,tau=tau,nstep=nstep)

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
  test_metropolis()
