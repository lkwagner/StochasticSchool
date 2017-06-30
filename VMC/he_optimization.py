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
    # Use your code to evaluate and store energies and errors.
    pass

  import pandas as pd
  pd.DataFrame(df).to_csv("helium.csv",index=False)
      

