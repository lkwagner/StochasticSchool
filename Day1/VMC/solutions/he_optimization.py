from slaterwf import ExponentSlaterWF
from wavefunction import JastrowWF,MultiplyWF
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
  for i in ['alpha','beta','acceptance']:
    df[i]=[]

  ham=Hamiltonian(Z=2) # Helium
  # Best Slater determinant.
  beta=0.0 # For book keeping.
  for alpha in np.linspace(1.5,2.5,11):
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
      df['beta'].append(beta)
      df['acceptance'].append(acc)

  # Best Slater-Jastrow.
  for alpha in np.linspace(1.5,2.5,11):
    for beta in np.linspace(-0.5,1.5,11):
      wf=MultiplyWF(ExponentSlaterWF(alpha=alpha),JastrowWF(a_ee=beta))
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
        df['beta'].append(beta)
        df['acceptance'].append(acc)

  import pandas as pd
  pd.DataFrame(df).to_csv("helium.csv",index=False)
      

