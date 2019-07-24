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
  poscur = pos.copy()
  wfold  = wf.value(poscur)
  acceptance=0.0
  nconf=pos.shape[2]
  # This loop performs the metropolis move many times to attempt to decorrelate the samples.
  for istep in range(nstep):
    # propose a move
    gauss_move_old = np.random.randn(*poscur.shape)
    posnew=poscur+np.sqrt(tau)*gauss_move_old

    wfnew=wf.value(posnew)

    # calculate Metropolis-Rosenbluth-Teller acceptance probability
    prob = wfnew**2/wfold**2 # for reversible moves

    # get indices of accepted moves
    acc_idx = (prob + np.random.random_sample(nconf) > 1.0)

    # update stale stored values for accepted configurations
    poscur[:,:,acc_idx] = posnew[:,:,acc_idx]
    wfold[acc_idx] = wfnew[acc_idx]
    acceptance += np.mean(acc_idx)/nstep

  return poscur,acceptance


##########################################   Test

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
  possample,acc = metropolis_sample(possample,wf,tau=tau,nstep=nstep)

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
  
