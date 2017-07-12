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
    return -2*self.alpha/dist + self.alpha**2
#-------------------------

if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  import wavefunction
  # 2 electrons, 3 dimensions, 5 configurations.
  testpos=np.random.randn(2,3,5)
  print("Exponent wavefunction")
  ewf=ExponentSlaterWF(0.5)
  wavefunction.test_wavefunction(ExponentSlaterWF(alpha=0.5))
  
