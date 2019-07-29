import numpy as np

print("Generate zeros:\n",np.zeros((2,2)))

a=np.zeros((2,2))
print("The shape of a matrix:\n",a.shape)

print("Generate random normal numbers:",
      np.random.randn(2,2))

print("Generate random uniform numbers in a 2x2 matrix:\n",
      np.random.random((2,2)))

print("Summing a vector:\n",np.sum(np.random.random(10)))

print("Summing over only one axis:\n",np.sum(np.random.random((10,10)),axis=1))


print("Taking the mean of a vector:\n",
      np.mean(np.random.random((50))))

print("Taking the standard deviation of a vector:\n",
    np.std(np.random.random((50))))

print("Square root:\n",np.sqrt(np.random.random(10)))
print("Exponential:\n",np.exp(np.random.random(10)))
print("Raise to a power:\n",np.random.random(10)**2)

print("\n#### Slicing")
a=np.random.random(5)
print("vector",a)
print("Simple slice",a[0:2],a[2:4])
print("Conditional slice: a[a>0.5] ",a[a>0.5])



