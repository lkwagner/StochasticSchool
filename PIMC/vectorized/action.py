import numpy as np

class KineticAction:
  def __init__(self,tau,lam):
    self.tau = tau
    self.lam = lam
  def kinetic_link_action(self,paths,islice,jslice):
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
    r2_arr = paths**2.
    r2 = r2_arr.sum(axis=1).sum(axis=1) # sum over ptcl and dim
    r2_mean = r2.mean(axis=0) # average over time slices
    return self.omega**2.*r2_mean/(4.*self.lam)
  def potential_action(self,paths,islice):
    nslice,nptcl,ndim,nconf = paths.shape
    r2_arr = paths**2.
    r2 = r2_arr.sum(axis=1).sum(axis=1) # sum over ptcl and dim
    inext = (islice+1)%nslice
    avg_r2 = 0.5*(r2[islice]+r2[inext])
    return self.tau*self.omega**2.*avg_r2/(4.*self.lam)

if __name__ == '__main__':
  pass
