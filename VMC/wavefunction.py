import numpy as np

class JastrowWF:
  """
  Jastrow factor of the form 

  exp(J_ee)

  J_ee = a_ee|r_1 - r_2| 

  """
  def __init__(self,a_ee):
    self.a_ee=a_ee
  #-------------------------

  def value(self,pos):
    eedist=np.sqrt(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0))
    exp_ee=(self.a_ee*eedist) #/(1 + self.eep_den*eedist)
    return np.exp(exp_ee)
  #-------------------------

  def gradient(self,pos):
    eedist=(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)[np.newaxis,:]
    # Partial derivatives of electron-electron distance.
    pdee=np.outer([1,-1],(pos[0,:,:]-pos[1,:,:])/eedist).reshape(pos.shape)
    grad_ee=self.a_ee*pdee 
    return grad_ee 
  #-------------------------

  def laplacian(self,pos):
    eedist=(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)[np.newaxis,:]
    pdee=np.outer([1,-1],(pos[0,:,:]-pos[1,:,:])/eedist).reshape(pos.shape)
    pdee2=pdee[0]**2 # Sign doesn't matter if squared.
    pd2ee=(eedist**2-(pos[0,:,:]-pos[1,:,:])**2)/eedist**3
    lap_ee=np.sum(self.a_ee*pd2ee + self.a_ee**2*pdee2,axis=0)
    #Laplacian is the same for both electrons
    return np.array([lap_ee,lap_ee])
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
  """ test """
  testpos=np.random.randn(2,3,5)
  df={'delta':[],
      'derivative err':[],
      'laplacian err':[]
      }
  for delta in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    df['delta'].append(delta)
    df['derivative err'].append(derivative_test(testpos,wf,delta))
    df['laplacian err'].append(laplacian_test(testpos,wf,delta))

  import pandas as pd
  df=pd.DataFrame(df)
  print(df)
  return df

########################################

if __name__=="__main__":
  import pandas as pd
  testpos=np.random.randn(2,3,5)

  print("Jastrow wavefunction")
  jas=JastrowWF(1.0)
  test_wavefunction(jas)

  print("Multiplied wavefunction")
  mwf=MultiplyWF(JastrowWF(1.0),JastrowWF(0.8))
  test_wavefunction(mwf)

