#!/bin/env python
import numpy
import pylab
from math import *

def Mean (myArray):
    return sum(myArray)/len(myArray)

def Var (myArray):
    mean = Mean(myArray)
    mean2 = Mean(myArray**2)
    N = len(myArray)+0.0
    return N/(N-1) * (mean2 - mean**2)

def NaiveStandardError(myArray):
  # calculate the standard error by calling the standard deviation function
  return sqrt(Var(myArray))/sqrt(len(myArray)+0.0)

def BinData(data, n):
    nbins = len(data)/n
    binned = numpy.zeros(nbins,float)
    # fill in the binned array here
    for i in range(0,nbins):
        binned[i] = numpy.average(data[n*i:(i+1)*n+1])
    return binned

def BinningMethod(data):
    errList = []
    for n in range(1,len(data)/20):
        errList.append(NaiveStandardError(BinData(data,n)))
    return errList

def C(data, i, mean, var):
    delta = data - mean
    N = len(data)
    return numpy.sum(delta[0:N-i]*delta[i:N])/((N-i)*var)
    
def Kappa(data):
    done = False
    N = len(data)
    i = 1
    mean = Mean (data)
    var  = Var  (data)
    csum = 1.0
    while ((not done) and i<N):
        c = C(data,i,mean,var)
        if (c < 0.0):
            done = True
        else:
            csum = csum + 2.0*c
        i = i+1
    return csum

def StdError(data):
    var = Var(data)
    kappa = Kappa(data)
    Neff = len(data)/kappa
    return sqrt(var/Neff)

def Stats(data):
    return (Mean(data), StdError(data), Kappa(data))
    
