import numpy as np
from wavefunction import JastrowWF,MultiplyWF
from slaterwf import ExponentSlaterWF
from hamiltonian import Hamiltonian
import pandas as pd

def check_singularity(wf,ham):
  ''' Check what happens to the energies when the two electrons in a wave
  function are positioned very close.'''

  nsamp=100
  pos=np.zeros((2,3,nsamp))
  pos[1,0,:]=1.0
  pos[0,0,:]=np.linspace(-0.501,1.399,nsamp)

  ke=-0.5*np.sum(wf.laplacian(pos),axis=0)
  vion=ham.pot_en(pos)
  vee=ham.pot_ee(pos)

  return pd.DataFrame({
      'pos':pos[0,0,:],
      'kinetic':ke,
      'electron-electron':vee,
      'electron-nucleus':vion
    })

def check_wfs():
  newdf=check_singularity(
      wf=ExponentSlaterWF(1.0),
      ham=Hamiltonian(Z=2)
    )
  newdf['wf']='slater unopt.'
  resdf=newdf

  resdf['electron-nucleus']

  newdf=check_singularity(
      wf=ExponentSlaterWF(2.0),
      ham=Hamiltonian(Z=2)
    )
  newdf['wf']='slater opt.'
  resdf=pd.concat([resdf,newdf])

  newdf=check_singularity(
      wf=MultiplyWF(ExponentSlaterWF(2.0),JastrowWF(0.5)),
      ham=Hamiltonian(Z=2)
    )
  newdf['wf']='slater-jastrow'
  resdf=pd.concat([resdf,newdf])

  return resdf

if __name__=='__main__':
  resdf=check_wfs()
  with open('singularity.csv','w') as outf:
    resdf.to_csv(outf)



