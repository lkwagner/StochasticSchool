import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
npts=50
#Generate some data
df={'a':np.random.random(npts),
    'b':np.random.randn(npts),
    'category':['cat1']*int(npts/2) + ['cat2']*int(npts/2)  #Using Python list generation
    }
df=pd.DataFrame(df)


#Scatter
plt.figure()
plt.scatter('a','b',data=df)
plt.xlabel('a')
plt.ylabel('b')
plt.savefig("scatter.pdf",bbox_inches='tight')

#Conditional plot
plt.figure()
sns.violinplot(x='category',y='b',data=df)

#Categorical plot with bootstrapped error bars
sns.factorplot(x='category',y='a',data=df)
sns.factorplot(x='category',y='b',data=df)

#Regression plot
plt.figure()
sns.regplot(x='a',y='b',data=df)


plt.show()
