import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("white")
df = pd.read_csv("dmc.csv")
df["step"] += 1
print(df.keys())

g=sns.PairGrid(df,x_vars=['step'],y_vars=['elocal','weight','eref','weightvar'],hue='tau')
g.map(plt.scatter,s=1)
g.add_legend()
plt.savefig("traces.pdf", bbox_inches='tight')