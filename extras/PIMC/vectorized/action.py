import numpy as np

# ==== you will need to fill in KineticAction and HarmonicPotentialAction ====
class KineticAction:
  def __init__(self,tau,lam):
    """ tau: time step i.e. imaginary time between adjacent beads
        lam: diffusion constant in imaginary time (lam = hbar^2/2m) """
    self.tau = tau
    self.lam = lam
  def kinetic_link_action(self,paths,islice,jslice):
    """ kinetic link action between imaginary time slice islice and jslice 
    Inputs:
      paths: 4D numpy array of shape (nslice,nptcl,ndim,nconf)
      islice: int, index of first time slice
      jslice: int, index of second time slice 
    Output:
      klink: 1D array of floats, kinetic link action, one for each configuration
    """
    klink = np.zeros(paths.shape[-1])
    return klink

class HarmonicPotentialAction:
  def __init__(self,tau,lam,omega):
    self.tau   = tau
    self.lam   = lam
    self.omega = omega
  def potential_energy(self,paths):
    """ V(x) = omega**2*x**2/(4*lam)
    Inputs:
      paths: 4D numpy array of shape (nslice,nptcl,ndim,nconf)
    Output:
      pot 
    """
    return 0.0
  def potential_link_action(self,paths,islice,jslice):
    """ potential link action between imaginary time slice islice and jslice 
    Inputs:
      paths: 4D numpy array of shape (nslice,nptcl,ndim,nconf)
      islice: int, index of first time slice
      jslice: int, index of second time slice 
    Output:
      plink: 1D array of floats, potential link action, one for each configuration
    """
    plink = np.zeros(paths.shape[-1])
    return plink
# ==== you will need to fill in KineticAction and HarmonicPotentialAction ====

def primitive_action(paths,omega,lam,tau):
  """ calculate the primitive action of paths in a harmonic trap
  Inputs:
    pahts: 4D numpy array of shape (nslice,nptcl,ndim,nconf), which stores an ensemble of paths in coordinate space
    omega: frequency of harmonic trap, potential V(x) = omega**2*x**2/(4*lam)
    lam: diffusion coefficient in imaginary time, lam=1/(2m)
    tau: inverse temperature i.e. imaginary time between adjacent time slices
  Output:
    action: 1D numpy array of shape (nconf,), which stores the actions of paths, one for each configuration
  """
  kaction = KineticAction(tau,lam)
  ext_pot = HarmonicPotentialAction(tau,lam,omega)
  action = 0.0
  nslice,nptcl,ndim,nconf = paths.shape
  for islice in range(nslice):
    inext   = (islice+1)%nslice
    action += kaction.kinetic_link_action(paths,islice,inext)
    action += ext_pot.potential_link_action(paths,islice,inext)
  return action

def exact_action(paths,omega,lam,beta):
  """ minus logarithm of linear harmonic oscillator density matrix diagonal:
     i.e. action = -log(rho)
    potential V(x) = 0.5*m*omega**2.*x**2 =omega**2*x**2/(4*lam) """
  nslice,nptcl,ndim,nconf = paths.shape
  assert ndim==1
  assert nslice==1
  ratio = omega/(4*lam*np.sinh(beta*omega))
  const = -0.5*np.log( ratio/np.pi )
  r2    = ((paths)**2.).sum(axis=0).sum(axis=0).sum(axis=0)
  exp   = -2.*r2*(np.cosh(beta*omega)-1)
  return -(ratio*exp + const)

def primitive_action_for_slice(paths,omega,lam,tau,islice):
  """ action related to slice islice """
  return 0.0

def test_actions():
  np.random.seed(0) 
  nslice = 3
  nptcl  = 2
  ndim   = 3
  nconf  = 2
  test_paths = np.random.randn(nslice,nptcl,ndim,nconf)

  tau = 0.1
  lam = 0.5
  omega = 1.0

  kaction = KineticAction(tau,lam)
  ext_pot = HarmonicPotentialAction(tau,lam,omega)

  pact = []
  kact = []
  for islice in range(nslice):
    inext = (islice+1)%nslice
    pact.append( ext_pot.potential_link_action(test_paths,islice,inext) )
    kact.append( kaction.kinetic_link_action(test_paths,islice,inext) )
  # end for
  pact_arr = np.array(pact).flatten()
  kact_arr = np.array(kact).flatten()
  #print ','.join( pact_arr.astype(str) )
  #print ','.join( kact_arr.astype(str) )
  pact_expect = np.array([0.471578396077,0.257973768519,0.470199767746,0.256899451026,0.423203984184,0.421086269681])
  kact_expect = np.array([41.7887726871,48.4422363557,33.5195343701,74.1256766531,13.6403857502,115.02306664])

  if not np.allclose(pact_arr,pact_expect):
    raise RuntimeError('potential link action is wrong')
  if not np.allclose(kact_arr,kact_expect):
    raise RuntimeError('kinetic link action is wrong')

def test_action_change():
  np.random.seed(0) 
  nslice = 3
  nptcl  = 2
  ndim   = 3
  nconf  = 2
  test_paths = np.random.randn(nslice,nptcl,ndim,nconf)

  tau = 0.1
  lam = 0.5
  omega = 1.0
  kaction = KineticAction(tau,lam)
  ext_pot = HarmonicPotentialAction(tau,lam,omega)

  move = np.random.randn(nptcl,ndim,nconf)

  old_action = primitive_action(test_paths,omega,lam,tau)
  old_action_for_slice0 =  primitive_action_for_slice(test_paths,omega,lam,tau,0)
  test_paths[0,:,:,:] += move
  new_action_for_slice0 =  primitive_action_for_slice(test_paths,omega,lam,tau,0)
  new_action = primitive_action(test_paths,omega,lam,tau)

  action_change1 = new_action_for_slice0 - old_action_for_slice0
  action_change2 = new_action - old_action
  if np.allclose(action_change1,action_change2):
    print('action change test passed!')
  else:
    raise RuntimeError('action change for a single slice is incorrect.')

if __name__ == '__main__':
  test_actions()
  #test_action_change()
