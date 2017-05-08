import numpy as np

#The wave function object. 
#This takes in a position and returns 
#either the value of the wave function or derivatives.
class Wavefunction:
  """ Positions are three index arrays as follows:
    [particle,dimension,configuration]
  """
  def __init__(self):
    pass
#-------------------------
  def value(self,x):
    pass
#-------------------------
  def gradient(self,x):
    """ Return grad psi/psi as a 3D array with dimensions
    [particle, dimension, configuration]
    """
    pass
#-------------------------
  def laplacian(self,x):
    pass
#-------------------------

########################################

class ExponentSlaterWF:
  def __init__(self,alpha=1):
    self.alpha=alpha
    pass
#-------------------------
  def value(self,x):
    r=np.sqrt(np.sum(x**2,axis=1))
    return np.exp(-self.alpha*r[0,:])*np.exp(-self.alpha*r[1,:])
#-------------------------
  def gradient(self,x):
    r=np.sqrt(np.sum(x**2,axis=1))
    return -self.alpha*x/r[:,np.newaxis,:]
#-------------------------
  def laplacian(self,x):
    r=np.sqrt(np.sum(x**2,axis=1))
    xoverr=x/r[:,np.newaxis,:]
    return np.sum(self.alpha**2 * xoverr**2 
        -self.alpha * (xoverr - xoverr**3)/x, 
        axis=1)
#-------------------------

########################################
class JastrowWF:
  def __init__(self):
    pass
#-------------------------
  def value(self,x):
    pass
#-------------------------
  def gradient(self,x):
    pass
#-------------------------
  def laplacian(self,x):
    pass
#-------------------------

########################################
class MultiplyWF:
  def __init__(self,wf1,wf2):
    pass
#-------------------------
  def value(self,x):
    pass
#-------------------------
  def gradient(self,x):
    pass
#-------------------------
  def laplacian(self,x):
    pass
#-------------------------

########################################

def derivativeTest(testpos,wf,delta=1e-4):
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
    
def laplacianTest(testpos,wf,delta=1e-5):
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

if __name__=="__main__":
  wf=ExponentSlaterWF(0.5)
  testpos=np.random.randn(2,3,5)
  df={'delta':[],
      'rms derivative':[],
      'rms laplacian':[]
      }
  for delta in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    df['delta'].append(delta)
    df['rms derivative'].append(derivativeTest(testpos,wf,delta))
    df['rms laplacian'].append(laplacianTest(testpos,wf,delta))

  import pandas as pd
  print(pd.DataFrame(df))

  
