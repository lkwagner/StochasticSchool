import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style('ticks')

#assert len(sys.argv)==2,"""
#  Error. Usage: python plot_csv.py sample.csv.
#  """

def plot_param(avgdf):
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
      ax.set_ylabel('Total Energy (Ha)')
    axis[-1].set_xlabel(r'$\%s$'%param)

  fig.set_size_inches(8,6)
  fig.tight_layout()
  fig.savefig('energy.pdf')

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


# Read and average over configurations.
#inpfn=sys.argv[1]
inpfn="helium.csv"
csvdf=pd.read_csv(open(inpfn,'r'))
csvdf['total'] = csvdf['electron-electron']+csvdf['electron-nucleus']+csvdf['kinetic']
csvdf['nonint'] = csvdf['electron-nucleus']+csvdf['kinetic']
avgdf=csvdf.groupby(['alpha','beta','acceptance']).apply(average_configs).reset_index()

plot_param(avgdf)

