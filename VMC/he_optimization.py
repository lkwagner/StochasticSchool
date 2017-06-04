from slaterwf import ExponentSlaterWF
from hamiltonian import Hamiltonian
from metropolis import metropolis_sample
import numpy as np

if __name__=="__main__":
  nconfig=1000
  ndim=3
  nelec=2
  nstep=100
  tau=0.2

  # All the quantities we will keep track of
  df={}
  quantities=['kinetic','electron-nucleus','electron-electron']
  for i in quantities:
    df[i]=[]
   # df[i+'-err']=[]
  for i in ['alpha','acceptance']:
    df[i]=[]

  ham=Hamiltonian(Z=2) # Helium
  for alpha in np.linspace(1.5,2.5,10):
    wf=ExponentSlaterWF(alpha=alpha)
    sample,acc = metropolis_sample(np.random.randn(nelec,ndim,nconfig),
                                   wf,tau=tau,nstep=nstep)
    
    ke=-0.5*np.sum(wf.laplacian(sample),axis=0)
    vion=ham.pot_en(sample)
    vee=ham.pot_ee(sample)
    
    for i in range(nconfig):
      for nm, quant in zip(quantities,
                           [ke,vion,vee]):
         df[nm].append(quant[i])
      df['alpha'].append(alpha)
      df['acceptance'].append(acc)

  import pandas as pd
  pd.DataFrame(df).to_csv("helium.csv",index=False)
      

