from slaterwf import ExponentSlaterWF
from wavefunction import JastrowWF,MultiplyWF
from hamiltonian import Hamiltonian
from metropolis import metropolis_sample
import numpy as np

if __name__=="__main__":
  # These are good parameters to start with.
  nconfig=1000
  ndim=3
  nelec=2
  nstep=100
  tau=0.2

  # All the quantities we will keep track of.
  # You'll want to populate thes lists.
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
    pass

  # Best Slater-Jastrow.
  for alpha in np.linspace(1.5,2.5,11):
    for beta in np.linspace(-0.5,1.5,11):
      pass

  import pandas as pd
  pd.DataFrame(df).to_csv("helium.csv",index=False)
      

