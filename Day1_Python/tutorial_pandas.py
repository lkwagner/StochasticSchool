import numpy as np
import pandas as pd

#First make a dictionary. They all should be the same length.
df={'a':np.random.random(50),
    'b':np.random.randn(50),
    'category':['cat1']*25 + ['cat2']*25  #Using Python list generation
    }
df=pd.DataFrame(df)
print(df)

# Save to CSV
df.to_csv("tutorial_pandas.csv",index=False)
# Read from CSV
df2=pd.read_csv("tutorial_pandas.csv")
print(df2)

#Manipulate columns; they act like numpy vectors
df['c']=df['a']+df['b']

#Groups
groups=df.groupby("category")
for nm,g in groups:
  print("Mean",nm, np.mean(g['a']))

#Reduction
averaged=df.groupby("category",as_index=False).mean().add_suffix("_mean").reset_index()
print(averaged)

#Compute errors 
def errorbar(a):
  std=a.std()
  n=a.count()
  return std/np.sqrt(n)

err=df.groupby("category").apply(errorbar).add_suffix("_err").reset_index()
print(err)

#Join the average and error together
avg_err=averaged.join(err)
#Remove some extra parts
avg_err=avg_err.drop(['category_mean','category_err'],1)
print(avg_err)
