import numpy as np


class ExponentSlaterWF:
  """ 
  Slater determinant specialized to one up and one down electron, each with
  exponential orbitals.
  Member variables:
    alpha: decay parameter.
  """
  def __init__(self,alpha=1):
    self.alpha=alpha
#-------------------------
  def value(self,pos):
    dist=np.sqrt(np.sum(pos**2,axis=1))
    return np.exp(-self.alpha*dist[0,:])*np.exp(-self.alpha*dist[1,:])
#-------------------------
  def gradient(self,pos):
    dist=np.sqrt(np.sum(pos**2,axis=1))
    return -self.alpha*pos/dist[:,np.newaxis,:]
#-------------------------
  def laplacian(self,pos):
    dist=np.sqrt(np.sum(pos**2,axis=1))
    pos_over_dist=pos/dist[:,np.newaxis,:]
    return np.sum(self.alpha**2 * pos_over_dist**2 
        -self.alpha * (pos_over_dist - pos_over_dist**3)/pos, 
        axis=1)
#-------------------------

