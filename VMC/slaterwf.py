import numpy as np


class ExponentSlaterWF:
  """ 
  Slater determinant specialized to one up and one down electron, each with
  exponential orbitals.
  Member variables:
    alpha: decay parameter.

  Note:
  pos is an array such that
    pos[i][j][k]
  will return the j-th component of the i-th electron for the 
  k-th sample (or "walker").
  """
  def __init__(self,alpha=1):
    self.alpha=alpha
#-------------------------
  def value(self,pos):
    pass # Implement me!
    return np.zeros(pos.shape[2])
#-------------------------
  def gradient(self,pos):
    pass # Implement me!
    return np.zeros(pos.shape)
#-------------------------
  def laplacian(self,pos):
    pass # Implement me!
    return np.zeros((pos.shape[0],pos.shape[2]))
#-------------------------


if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  import wavefunction
  testpos=np.random.randn(2,3,5)
  print("Exponent wavefunction")
  ewf=ExponentSlaterWF(0.5)
  wavefunction.test_wavefunction(ExponentSlaterWF(0.5))
  
