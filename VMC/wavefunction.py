import numpy as np

#The wave function object. 
#This takes in a position and returns 
#either the value of the wave function or derivatives.
class Wavefunction:
  """ 
  Skelton class for wavefunctions.
  Positions are three index arrays as follows:
    [particle,dimension,configuration]
  """
  def __init__(self):
    pass
#-------------------------
  def value(self,pos):
    pass
#-------------------------
  def gradient(self,pos):
    """ Return grad psi/psi as a 3D array with dimensions
    [particle, dimension, configuration]
    """
    pass
#-------------------------
  def laplacian(self,pos):
    pass
#-------------------------

########################################

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

########################################
class JastrowWF:
  def __init__(self):
    pass
#-------------------------
  def value(self,pos):
    pass
#-------------------------
  def gradient(self,pos):
    pass
#-------------------------
  def laplacian(self,pos):
    pass
#-------------------------

########################################
class MultiplyWF:
  """ Wavefunction defined as the product of two other wavefunctions."""
  def __init__(self,wf1,wf2):
    self.wf1=wf1
    self.wf2=wf2
#-------------------------
  def value(self,pos):
    return self.wf1.value(pos)*self.wf2.value(pos)
#-------------------------
  def gradient(self,pos):
    return self.wf1.gradient(pos) + self.wf2.gradient(pos)
#-------------------------
  def laplacian(self,pos):
    return self.wf1.laplacian(pos) +\
           2*np.sum(self.wf1.gradient(pos)*self.wf2.gradient(pos),axis=1) +\
           self.wf2.laplacian(pos)
#-------------------------

########################################

def derivative_test(testpos,wf,delta=1e-4):
  """ Compare numerical and analytic derivatives. """
  wf0=wf.value(testpos)
  grad0=wf.gradient(testpos)
  npart=testpos.shape[0]
  ndim=testpos.shape[1]
  grad_numeric=np.zeros(grad0.shape)
  for p in range(npart):
    for d in range(ndim):
      shift=np.zeros(testpos.shape)
      shift[p,d,:]+=delta
      wfval=wf.value(testpos+shift)
      grad_numeric[p,d,:]=(wfval-wf0)/(wf0*delta)
  
  return np.sqrt(np.sum((grad_numeric-grad0)**2)/(npart*testpos.shape[2]*ndim))

########################################
    
def laplacian_test(testpos,wf,delta=1e-5):
  """ Compare numerical and analytic Laplacians. """
  wf0=wf.value(testpos)
  lap0=wf.laplacian(testpos)
  npart=testpos.shape[0]
  ndim=testpos.shape[1]
  
  lap_numeric=np.zeros(lap0.shape)
  for p in range(npart):
    for d in range(ndim):
      shift=np.zeros(testpos.shape)
      shift[p,d,:]+=delta      
      wf_plus=wf.value(testpos+shift)
      shift[p,d,:]-=2*delta      
      wf_minus=wf.value(testpos+shift)
      # Here we use the value so that the laplacian and gradient tests
      # are independent
      lap_numeric[p,:]+=(wf_plus+wf_minus-2*wf0)/(wf0*delta**2)
  
  return np.sqrt(np.sum((lap_numeric-lap0)**2)/(npart*testpos.shape[2]))
########################################

def test_wavefunction(wf):
  """ Convenience function for running several tests on a wavefunction. """
  testpos=np.random.randn(2,3,5)
  df={'delta':[],
      'derivative':[],
      'laplacian':[]
      }
  for delta in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    df['delta'].append(delta)
    df['derivative'].append(derivative_test(testpos,wf,delta))
    df['laplacian'].append(laplacian_test(testpos,wf,delta))

  import pandas as pd
  print("RMS differences")
  print(pd.DataFrame(df))
########################################

if __name__=="__main__":
  import pandas as pd
  testpos=np.random.randn(2,3,5)

  print("Exponent wavefunction")
  ewf=ExponentSlaterWF(0.5)

  test_wavefunction(ExponentSlaterWF(0.5))
  mwf=MultiplyWF(ExponentSlaterWF(0.5),ExponentSlaterWF(0.6))
  print("Multiplied wavefunction")
  test_wavefunction(mwf)
