import numpy as np
import matplotlib.pyplot as plt

class DBSCAN():
    def __init__(self,file,data,eps,MinPts):
        self.file=file
        self.data=data
        self.eps=eps
        self.MinPts=MinPts
        self.indices=np.array([i for i in range(self.data.shape[0])])
        self.status=np.array(["X" for i in range(self.data.shape[0])]) # X:non-visited, C:core, E:edge, O:outlier
        self.label=np.array([-1 for i in range(self.data.shape[0])]) # -1:noise, i:cluster i
        self.distance=np.array([[self.Euclidean(x,y) for y in self.data]for x in self.data])

    def Euclidean(self,x,y):
        return np.sum(np.power(x-y,2))**(1/2)

    def SearchNeighbor(self,i):
        isNeighbor=(self.distance[i]<=self.eps)
        return self.indices[isNeighbor].tolist()

    def Visualization(self):
        self.iter+=1
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        for i in range(np.max(self.label)+1):
            cluster=self.data[(self.label==i)]
            X=[d[0] for d in cluster]
            Y=[d[1] for d in cluster]
            plt.plot(X,Y,color=color[i],linestyle='none',marker='o')
        unknown=self.data[(self.status=="X")]
        X=[d[0] for d in unknown]
        Y=[d[1] for d in unknown]
        plt.plot(X,Y,color='gray',linestyle='none',marker='o')
        outlier=self.data[(self.status=="O")]
        X=[d[0] for d in outlier]
        Y=[d[1] for d in outlier]
        plt.plot(X,Y,color='k',linestyle='none',marker='x')
        plt.savefig("DBSCAN_{}_{}_{}_{}.png".format(self.file,self.eps,self.MinPts,self.iter))
        plt.close('all')

    def CombineList(self,X,Y):
        d=[x for x in X]
        for y in Y:
            d.append(y)
        return d

    def run(self):
        self.iter=0
        for i in range(self.data.shape[0]):
            if self.status[i]=="X":
                index=self.SearchNeighbor(i)
                if len(index)<self.MinPts:
                    self.status[i]="O"
                    self.label[i]=-1
                else:
                    C=(np.max(self.label)+1)
                    self.status[i]="C"
                    self.label[i]=C
                    j=0
                    while(j!=len(index)):
                        if self.status[index[j]]=="O":
                            self.status[index[j]]="E"
                            self.label[index[j]]=C
                        elif self.status[index[j]]=="X":
                            _index=self.SearchNeighbor(index[j])
                            if len(_index)<self.MinPts:
                                self.status[index[j]]="E"
                                self.label[index[j]]=C
                            else:
                                self.status[index[j]]="C"
                                self.label[index[j]]=C
                                index=self.CombineList(index,_index)
                                self.Visualization()
                        j+=1
        self.Visualization()