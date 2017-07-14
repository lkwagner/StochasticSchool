#!/usr/bin/env python
import numpy as np

##########################################   Path
class Path:
  def __init__(self,beads,tau,lam,str_rep=False):
    """ constructor for the path class
    Inputs:
      beads: 3D numpy array of shape (nslice,nptcl,ndim)
      tau: imaginary time between neighboring time slices 
      lam: diffusion coefficient in imaginary time (in atomic units: 1/2m), where m is the mass of the particles - we assume a single species of identical particles 
      str_rep: boolean, print a string representation after initialization
    Outputs:
      none
    Effects:
      constructs the Path class, also prints a string reprensentation of itself if str_rep """
    
    self.nslice,self.nptcl,self.ndim = beads.shape
    self.tau    = tau
    self.lam    = lam
    self.beads  = beads.copy()
    self.beta   = self.tau*self.nslice
    
    self.NumTimeSlices = self.nslice
    self.NumParticles  = self.nptcl
    
    if str_rep:
      print(self)

  def __str__(self):
    rep = "path nptcl = %d, inverse temperature beta = %6.4f" %\
      (self.nptcl,self.beta)
    return rep

  @staticmethod
  def draw_beads_3d(ax,beads):
    """ draw all beads in 3D
    Inputs:
     ax: matplotlib.Axes3D object
     beads: 3D numpy array of shape (nslice,nptcl,ndim)
    Output:
     ptcls: a list of pairs of plot objects. There is ony entry for each particle. Each entry has two items: line representing the particle and text labeling the particle.
    Effect:
     draw all particles on ax """

    nslice,nptcl,ndim = beads.shape
    com = beads.mean(axis=0) # center of mass of each particle, used to label the particles only

    ptcls = []
    for iptcl in range(nptcl):
      mypos = beads[:,iptcl,:] # all time slices for particle iptcl
      pos = np.insert(mypos,0,mypos[-1],axis=0) # close beads

      line = ax.plot(pos[:,0],pos[:,1],pos[:,2],marker='o') # draw particle
      text = ax.text(com[iptcl,0],com[iptcl,1],com[iptcl,2],'ptcl %d' % iptcl,fontsize=20) # label particle
      ptcls.append( (line,text) )
    return ptcls

  def SetCouplingConstant(self,c):
    self.c=c
  def SetPotential(self,externalPotentialFunction):
    self.VextHelper=externalPotentialFunction
  def Vee(self,pos):
    # you will write this
    # using self.c
    return 0.0
  def Vext(self,pos):
    return self.VextHelper(pos)
  def KineticAction(self,slice1,slice2):
    if slice1<0 or slice1>self.nslice-1 or slice2<0 or slice2>self.nslice-1:
      raise RuntimeError('time slice out of bounds')
    # you will fill this in
    tot=0.0

    # ---- solution ----
    sq_diff = (self.beads[slice1] - self.beads[slice2])**2.
    numerator   = sq_diff.sum() # sum over particles and dimensions
    denominator = 4.*self.lam*self.tau
    tot = numerator/denominator
    # ---- solution ----

    return tot
  def PotentialAction(self,slice1,slice2):
    if not abs(slice1-slice2)==1:
      raise RuntimeError('symmetrized primitive potential action must involve neighboring imaginary time slices')
    # you will fill this in
    pot = 0.0

    # ---- solution ----
    pot = 0.5*self.tau*( self.Vext(self.beads[slice1]) + self.Vext(self.beads[slice2]) )
    # ---- solution ----
    return pot
  def RelabelBeads(self):
    slicesToShift=np.random.randint(0,self.NumTimeSlices-1)
    l=range(slicesToShift,len(self.beads))+range(0,slicesToShift)
    self.beads=self.beads[l].copy()
  def KineticEnergy(self):
    # computes kinetic energy
    KE=0.0
    # ---- solution ----
    # thermodynamic estimator
    KE = self.ndim*self.nptcl/(2.*self.tau)
    for islice in range(self.nslice):
      sq_diff = (self.beads[islice] - self.beads[(islice-1)%self.nslice])**2.
      numerator   = sq_diff.sum() # sum over particles and dimensions
      denominator = 4.*self.lam*self.tau*self.tau  #**2.
      KE -= numerator/denominator/self.nslice
    # ---- solution ----
    return float(KE)
  def PotentialEnergy(self):
    # computes potential energy
    PE=0.0
    # ---- solution ----
    for islice in range(self.nslice):
      for iptcl in range(self.nptcl):
        PE += self.Vext(self.beads[islice,iptcl,:])
    # ---- solution ----
    return float(PE)/self.nslice
  def Energy(self):
    return self.PotentialEnergy()+self.KineticEnergy()

##########################################   Test
def load_test_path(fname='data/TestPath.dat',nslice=5,nptcl=2,ndim=3
 ,tau=0.5,lam=0.5,str_rep=False):
  test_beads = np.loadtxt(fname).reshape(nslice,nptcl,ndim)
  path = Path(test_beads,tau,lam,str_rep=str_rep)
  return path

def test_initialization():
  tau = lam = 0.5
  path = load_test_path(str_rep=True)
  assert np.isclose(path.tau,tau)
  assert np.isclose(path.lam,lam)

  test_pos = path.beads[0,1,2]
  expect_pos = -0.566223936211
  assert np.isclose(test_pos,expect_pos)

def test_visualization():
  path = load_test_path()

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D # enable 3D
  fig = plt.figure()
  ax  = fig.add_subplot(1,1,1,projection='3d',aspect=1)
  ax.set_xlabel('x',fontsize=16)
  ax.set_ylabel('y',fontsize=16)
  ax.set_zlabel('z',fontsize=16)

  path.draw_beads_3d(ax,path.beads)

  fig.tight_layout()
  fig.savefig('figures/test_beads.png',dpi=300)
  plt.show()

def test_kinetic_action():
  path = load_test_path()
  kact = path.KineticAction(1,2)
  kact_expect = 0.567148772825
  assert np.isclose(kact,kact_expect)

def test_harmonic_potential():
  from action import HarmonicOscillator
  path = load_test_path()
  path.SetPotential(HarmonicOscillator)

  test_pos = [0.1,0.3,0.1]
  pot = path.Vext(np.array(test_pos))
  expect_pot = 0.5*np.dot(test_pos,test_pos)
  assert np.isclose(pot,expect_pot)

def test_kinetic_energy():
  path = load_test_path()
  ke = path.KineticEnergy()
  print(ke)

if __name__ == '__main__':
  print('testing initialization ...')
  test_initialization()
  print('initialization test passed')

  print('\ntesting kinetic action ...')
  test_kinetic_action()
  print('kinetic action test passed')

  print('\ntesting visualization ...')
  test_visualization()

  print('\ntesting harmonic potential ...')
  test_harmonic_potential()
  print('harmonic potential test passed')

  print('\ntesting kinetic energy ...')
  test_kinetic_energy()
