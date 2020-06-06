from random import uniform
import matplotlib.pyplot as plt
import numpy as np

class KMeans():
    def __init__(self,file,data,k):
        self.k=k
        self.file=file
        self.data=data
        mini=np.min(data,axis=0)
        maxi=np.max(data,axis=0)
        self.center=np.array([[uniform(mini[0],maxi[0]),uniform(mini[1],maxi[1])] for i in range(self.k)])

    def Euclidean(self,x,y):
        return np.sum(np.power(x-y,2))**(1/2)

    def Converge(self,x,y):
        return (np.sum(np.abs(x-y)) < 0.001)

    def Clustering(self):
        self.cluster=[[] for i in range(self.k)]
        for d in self.data:
            distance=np.array([self.Euclidean(d,c) for c in self.center ])
            self.cluster[np.argmin(distance)].append(d)

    def UpdateCenter(self):
        center=[]
        for i in range(self.k):
            center.append(np.mean(self.cluster[i],0))
        return np.array(center)

    def Visualization(self):
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        centercolor=['#0066cc','#ff0080','#6f00d2','#a23400','#ff0000']
        X=[[point[0] for point in c] for c in self.cluster]
        Y=[[point[1] for point in c] for c in self.cluster]
        for i in range(self.k):
            plt.plot(X[i],Y[i],color=color[i],linestyle='none',marker='o')
            plt.plot([self.center[i][0]],[self.center[i][1]],color=centercolor[i],linestyle='none',marker='^')
        plt.savefig('KMeans_{}_{}_{}.png'.format(self.file,self.k,self.iteration))
        plt.close('all')

    def run(self):
        self.iteration=0
        while 1:
            self.Clustering()
            self.Visualization()
            new=self.UpdateCenter()
            if self.Converge(self.center,new):
                break
            self.center=new
            self.iteration+=1