import numpy as np
import pandas as pd
from slaterwf import ExponentSlaterWF
from wavefunction import MultiplyWF,JastrowWF
from hamiltonian import Hamiltonian
from metropolis import metropolis_sample
from metropolis_drift import metropolis_sample_biased

def compare_drift(wf,ham):
  nelec=2
  ndim=3
  nconfig=1000
  nstep=100
  tau=0.1

  sample_met,acc_met = metropolis_sample(np.random.randn(nelec,ndim,nconfig),
                                 wf,tau=tau,nstep=nstep)

  sample_bia,acc_bia = metropolis_sample_biased(np.random.randn(nelec,ndim,nconfig),
                                 wf,tau=tau,nstep=nstep)

  ke_met=-0.5*np.sum(wf.laplacian(sample_met),axis=0)
  vion_met=ham.pot_en(sample_met)
  vee_met=ham.pot_ee(sample_met)

  ke_bia=-0.5*np.sum(wf.laplacian(sample_bia),axis=0)
  vion_bia=ham.pot_en(sample_bia)
  vee_bia=ham.pot_ee(sample_bia)

  res=pd.DataFrame({
      'method':['unbiased','biased'],
      'acceptance':[acc_met,acc_bia],
      'kinetic':[ke_met.mean(),ke_bia.mean()],
      'electron-electron':[vee_met.mean(),vee_bia.mean()],
      'electron-nucleus':[vion_met.mean(),vion_bia.mean()],
    })

  print(res)

ham=Hamiltonian(Z=2)

wf=MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5))
compare_drift(wf,ham)

