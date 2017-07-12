import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style('ticks')

#assert len(sys.argv)==2,"""
#  Error. Usage: python plot_csv.py sample.csv.
#  """

def average_configs(df):
  res=pd.Series({
      'electron-electron':df['electron-electron'].mean(),
      'electron-nucleus':df['electron-nucleus'].mean(),
      'kinetic':df['kinetic'].mean(),
      'nonint':df['nonint'].mean(),
      'total':df['total'].mean(),
      'electron-electron_err':df['electron-electron'].std()/len(df['electron-electron'])**0.5,
      'electron-nucleus_err':df['electron-nucleus'].std()/len(df['electron-nucleus'])**0.5,
      'kinetic_err':df['kinetic'].std()/len(df['kinetic'])**0.5,
      'nonint_err':df['nonint'].std()/len(df['nonint'])**0.5,
      'total_err':df['total'].std()/len(df['total'])**0.5
    })
  return res

def plot_optimization():
  ''' Plot out the optimizations for each of the wave functions and Hamiltonians.'''

  # Take appropriate averages.
  inpfn="helium.csv"
  csvdf=pd.read_csv(open(inpfn,'r'))
  csvdf['total'] = csvdf['electron-electron']+csvdf['electron-nucleus']+csvdf['kinetic']
  csvdf['nonint'] = csvdf['electron-nucleus']+csvdf['kinetic']
  avgdf=csvdf.groupby(['alpha','beta','acceptance']).apply(average_configs).reset_index()

  # Make plots.
  fig,axes=plt.subplots(2,2,sharex='col')

  for axis,param,sel in zip(
      axes.T,
      ['alpha','beta'],
      [avgdf['beta']==0.0,avgdf['alpha']==2.0]):
    df=avgdf[sel]
    for ax,prop,color in zip(axis,['nonint','total'],['b','r']):
      ax.errorbar(df[param],df[prop],df['%s_err'%prop],color=color,
        capthick=1,capsize=2)
      ax.plot(df[param],df[prop],color=color)
      ax.set_ylabel('%s energy (Ha)'%prop)
    axis[-1].set_xlabel(param)
  axes[0,0].set_title('Slater (beta=0)')
  axes[0,1].set_title('Slater-Jastrow (fixing alpha=2.0)')

  fig.set_size_inches(8,6)
  fig.tight_layout()
  fig.savefig('energy.pdf')

def plot_singularity():
  ''' Demonstrate how you can satisfy cusp conditions.'''

  csvdf=pd.read_csv(open('singularity.csv','r'))
  csvdf['total']=csvdf['kinetic']+csvdf['electron-electron']+csvdf['electron-nucleus']
  fig,axes=plt.subplots(3,1,sharex=True)
  for ax,wf in zip(axes,('slater unopt.','slater opt.','slater-jastrow')):
    pltdf=csvdf[csvdf['wf']==wf]

    ax.plot(pltdf['pos'],pltdf['kinetic'],'r',label='kinetic')
    ax.plot(pltdf['pos'],pltdf['electron-electron'],'b',label='e-e')
    ax.plot(pltdf['pos'],pltdf['electron-nucleus'],'g',label='e-n')
    ax.plot(pltdf['pos'],pltdf['total'],'k',label='total')
    ax.set_xlim((-0.6,1.6))
    ax.set_ylim((-100,100))
    ax.set_ylabel('Energy (Ha)')
    ax.set_title(wf)
    ax.legend(loc='best')

  axes[-1].set_xlabel('x (Angstroms)')
  fig.set_size_inches(4,9)
  fig.tight_layout()
  fig.savefig('slater_singularity.pdf')

plot_optimization()
#plot_singularity()
