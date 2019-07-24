from numpy import *
import random
import numpy
import pylab
from math import *


class Histogram:
        min=0.0
        max=5.0
        delta=0.05
        numSamples=0
        bins=numpy.array([0])
        def add(self,val):
                if self.min<val and val<self.max:
                        index=int((val-self.min)/self.delta)
                        self.bins[index]=self.bins[index]+1
                self.numSamples=self.numSamples+1
        def printMe(self):
                for i in range(0,len(self.bins)):
                        print self.min+self.delta*i,self.bins[i]/(self.numSamples+0.0)

        def plotMe(self,fileName=""):
                print "plotting"
                pylab.clf()
                self.bins=self.bins/self.numSamples
                xCoord=[self.min+self.delta*i for i in range(0,len(self.bins))]
                pylab.plot(xCoord,self.bins)
                pylab.gca().xaxis.major.formatter.set_scientific(False)
                if not(fileName==""):
                   pylab.savefig(fileName)
                else:
                   pylab.show()
        def plotMeNorm(self,fileName):
                print "plotting"
                pylab.clf()
                self.bins=self.bins/self.numSamples
                xCoord=numpy.array([self.min+self.delta*i for i in range(0,len(self.bins))])
                pylab.plot(xCoord,self.bins/(xCoord*xCoord+0.0001))
                pylab.gca().xaxis.major.formatter.set_scientific(False)
                pylab.savefig(fileName)
                pylab.show()

        def __init__(self,min,max,numBins):
                self.min=min
                self.max=max
                self.delta=(max-min)/(numBins+0.0)
                self.bins=numpy.zeros(numBins)+0.0
                numSamples=0

