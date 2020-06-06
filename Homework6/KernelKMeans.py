import numpy as np
import matplotlib.pyplot as plt
from math import exp

class KernelKMeans():
    def __init__(self,file,data,k,gamma,enhance):
        self.k=k
        self.gamma=gamma
        self.file=file
        self.data=data
        self.gram_matrix=self.RBFKernel(self.data,self.data)
        self.label=np.random.randint(k,size=self.data.shape[0])
        if enhance:
            if file=="circle":
                mask1 = (self.data[:, 0]**2 + self.data[:, 1]**2 < 0.01)
                mask2 = (self.data[:, 0]**2 + self.data[:, 1]**2 > 0.99)
            elif file=="moon":
                mask1 = (self.data[:, 1] > 0.75)
                mask2 = (self.data[:, 1] < -0.15)
            self.label[mask1] = 0
            self.label[mask2] = 1

    def Euclidean(self,x,y):
        return np.sum(np.power(x-y,2))**(1/2)

    def Gaussian(self,x,y):
        return exp((self.Euclidean(x,y)** 2)*(-1)*(self.gamma))

    def RBFKernel(self,X,Y):
        return np.array([[self.Gaussian(X[i],Y[j]) for j in range(Y.shape[0])] for i in range(X.shape[0])])

    def ComputeDistance(self):
        distance=np.zeros((self.data.shape[0],self.k))
        for i in range(self.k):
            iscluster=(self.label==i)
            num=iscluster.sum()
            cluster=self.gram_matrix[iscluster][:,iscluster]
            dist=np.sum(cluster/(num**2))
            distance[:, i]+=dist-(2*np.sum(self.gram_matrix[:,iscluster],axis=1)/num)
        return distance.argmin(axis=1)

    def Visualization(self,iteration):
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        for i in range(self.k):
            cluster=(self.label==i)
            X=[x[0] for x in self.data[cluster]]
            Y=[x[1] for x in self.data[cluster]]
            plt.plot(X,Y,color=color[i],linestyle='none',marker='o')
        plt.savefig('KernelKMeans_{}_{}_{}_{}.png'.format(self.file,self.k,self.gamma,iteration))
        plt.close('all')

    def Converge(self,x,y):
        return (np.sum(np.abs(x-y))==0)

    def run(self):
        for i in range(1000):
            label=self.ComputeDistance()
            self.Visualization(i)
            if self.Converge(self.label,label):
                break
            self.label=label