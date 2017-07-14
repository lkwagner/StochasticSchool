import numpy as np

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
      klink: float, kinetic link action
    """
    r2_arr = (paths[islice] - paths[jslice])**2. # (nptcl,ndim,nconf)
    r2 = r2_arr.sum(axis=0).sum(axis=0) # sum over ptcl, then dim
    return r2/(4.*self.lam*self.tau)
  def kinetic_action(self,paths,islice):
    nslice,nptcl,ndim,nconf = paths.shape
    tot = 0.0 

    iprev = (islice-1)%nslice
    tot  += self.kinetic_link_action(paths,iprev,islice)

    inext = (islice+1)%nslice
    tot  += self.kinetic_link_action(paths,islice,inext)

    return tot

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
    r2_arr = paths**2.
    r2 = r2_arr.sum(axis=1).sum(axis=1) # sum over ptcl and dim
    r2_mean = r2.mean(axis=0) # average over time slices
    return self.omega**2.*r2_mean/(4.*self.lam)
  def potential_link_action(self,paths,islice,jslice):
    """ potential link action between imaginary time slice islice and jslice 
    Inputs:
      paths: 4D numpy array of shape (nslice,nptcl,ndim,nconf)
      islice: int, index of first time slice
      jslice: int, index of second time slice 
    Output:
      plink: float, potential link action
    """
    nslice,nptcl,ndim,nconf = paths.shape
    r2_arr = paths**2.
    r2 = r2_arr.sum(axis=1).sum(axis=1) # sum over ptcl and dim
    avg_r2 = 0.5*(r2[islice]+r2[jslice])
    return self.tau*self.omega**2.*avg_r2/(4.*self.lam)

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
  """ action relatex to slice islice """
  return 0.0

if __name__ == '__main__':
  pass
