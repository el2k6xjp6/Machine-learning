import numpy as np
from math import exp
import matplotlib.pyplot as plt
from random import uniform

class SpectralClustering():
    def __init__(self,file,data,k,gamma):
        self.data=data
        self.file=file
        self.k=k
        self.gamma=gamma

    def Euclidean(self,x,y):
        return np.sum(np.power(x-y,2))**(1/2)

    def Gaussian(self,x,y):
        return exp((self.Euclidean(x,y)** 2)*(-1)*self.gamma)

    def GraphLaplacian(self,X,Y):
        #Adjancency matrix
        W=np.array([[self.Gaussian(X[i],Y[j]) for j in range(Y.shape[0])] for i in range(X.shape[0])])
        #Degree matrix
        D=np.diag(W.sum(axis=1))
        return D-W

    def Converge(self,x,y):
        if x.shape!=y.shape:
            return True
        return np.sum(np.abs(x-y)) < 0.001

    def Clustering(self,data):
        self.cluster=[[] for i in range(self.k)]
        label=[]
        for d in range(data.shape[0]):
            distance=np.array([self.Euclidean(data[d],c) for c in self.center ])
            self.cluster[np.argmin(distance)].append(data[d])
            label.append(np.argmin(distance))
        self.label=np.array(label)

    def UpdateCenter(self):
        center=[]
        for i in range(self.k):
            center.append(np.mean(self.cluster[i],0))
        return np.array(center)

    def Visualization(self):
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        for i in range(self.k):
            cluster=self.data[(self.label==i)]
            X=[d[0] for d in cluster]
            Y=[d[1] for d in cluster]
            plt.plot(X,Y,color=color[i],linestyle='none',marker='o')
        plt.savefig('SpectralClustering_{}_{}_{}_{}.png'.format(self.file,self.k,self.gamma,self.iteration))
        plt.close('all')

    def KMeans(self,k,data):
        self.k=k
        mini=np.min(data,axis=0)
        maxi=np.max(data,axis=0)
        self.center=np.array([[uniform(mini[i],maxi[i]) for i in range(data.shape[1]) ] for j in range(self.k)])
        self.iteration=0
        for i in range(1000):
            self.Clustering(data)
            self.Visualization()
            new=self.UpdateCenter()
            if self.Converge(self.center,new):
                break
            self.center=new
            self.iteration+=1

    def run(self):
        L=self.GraphLaplacian(self.data,self.data)
        evals,evecs=np.linalg.eig(L)
        evecs=evecs[:,np.argsort(evals)]
        evals=evals[np.argsort(evals)]
        self.KMeans(self.k,evecs[:,1:self.k])