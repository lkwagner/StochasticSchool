import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df={'x':[],'N':[]}
M=100
for N in range(1,10):
  X=np.mean(np.random.random((M,N)),axis=1)
  df['x'].extend(X)
  df['N'].extend([N]*M)

df=pd.DataFrame(df)

sns.set_style("white")
plt.figure(figsize=(4,2))
sns.violinplot(x='N',y='x',data=df,cut=0,inner='stick')
plt.ylabel("Observations of $S_N$")
sns.despine()
plt.savefig("clt.pdf",bbox_inches='tight')

  
