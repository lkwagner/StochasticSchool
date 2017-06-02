
import numpy as np


class Hamiltonian:
  def __init__(self,Z=2):
    self.Z=Z
    pass
  def pot_en(self,pos):
    """ electron-nuclear potential of configurations 'pos' """
    r=np.sqrt(np.sum(pos**2,axis=1))
    return np.sum(-self.Z/r,axis=0)
  def pot_ee(self,pos):
    """ electron-electron potential of configurations 'pos' """
    ree=np.sqrt(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0))
    return 1/ree
  def pot(self,pos):
    """ potential energy of configuations 'pos' """
    return self.pot_en(pos)+self.pot_ee(pos)

