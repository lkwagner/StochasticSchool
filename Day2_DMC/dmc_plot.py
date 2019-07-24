import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pyblock
sns.set_style("white")
df=pd.read_csv("dmc.csv")
df['step']+=1

groups=df.groupby('tau')
warmup=1000

df_reblock={'energy':[],
            'tau':[],
            'err':[]
            }

for nm,g in groups:
  reblock_data = pyblock.blocking.reblock(g['elocal'].values[warmup:])
  ind=pyblock.blocking.find_optimal_block(g['elocal'].values[warmup:].shape[0],reblock_data)
  print(ind[0])
  print(reblock_data[ind[0]])
  avgd=reblock_data[ind[0]]
  df_reblock['energy'].append(float(avgd.mean))
  df_reblock['tau'].append(nm)
  df_reblock['err'].append(float(avgd.std_err))


df_reblock=pd.DataFrame(df_reblock)
  

g=sns.FacetGrid(data=df,hue='tau')

g.map(plt.scatter, 'step','elocal',s=2,alpha=0.5)
plt.legend(loc=(1.05,0.5))
plt.savefig("elocal.pdf",bbox_inches='tight')

sns.lmplot(x='step',y='weight',data=df,scatter_kws={'s':1},hue='tau',legend=False)
plt.legend(loc=(1.05,0.5))
plt.savefig("weight.pdf",bbox_inches='tight')


sns.lmplot(x='step',y='eref',data=df,scatter_kws={'s':1},hue='tau',legend=False)
plt.legend(loc=(1.05,0.5))
plt.savefig("eref.pdf",bbox_inches='tight')


plt.figure()
plt.axhline(-2.903724)
plt.errorbar(x='tau',y='energy',yerr='err',data=df_reblock,
             marker='o',mew=1,mec='k')
plt.xlabel(r"$\tau$ (Hartree$^{-1}$)")
plt.ylabel("Energy(Hartree)")
plt.savefig("avg_vs_tau.pdf",bbox_inches='tight')
