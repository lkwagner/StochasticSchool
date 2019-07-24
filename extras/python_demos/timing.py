from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
df={'n':[],'t':[]}
for nexp in range(2,16):
  n=2**nexp
  start = timer()
  a=np.random.random(n)
  b=np.random.randn(n)
  np.dot(a,b)
  end = timer()
  df['n'].append(n)
  df['t'].append(1e6*(end-start))

df=pd.DataFrame(df)
f, ax = plt.subplots(figsize=(3,3))
ax.set(xscale="log", yscale="log")
sns.regplot("n", "t",df , ax=ax, 
            scatter_kws={"s": 40,'edgecolors':'k','linewidths':1})
plt.ylabel("Time ($\mu$s)")
plt.xlabel("n")
sns.despine()
plt.savefig("timing.pdf",bbox_inches='tight')
