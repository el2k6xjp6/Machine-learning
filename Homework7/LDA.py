import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt

class LinearDiscriminantAnalysis():
    def __init__(self,data,label):
        self.label=label
        Xj=np.array([data[(label==i)] for i in range(1,6)])
        Mj=np.array([ml.repmat(np.mean(data[(label==i)],axis=0)/np.sum(label==i),np.sum(label==i),1) for i in range(1,6)])
        C=Xj-Mj
        Sj=np.zeros([5,data.shape[1],data.shape[1]])
        for i in range(C.shape[0]):
            for j in C[i]:
                Sj[i]+=(np.matmul(np.array([j]).T,np.array([j])))
        Sw=np.zeros([data.shape[1],data.shape[1]])
        for i in Sj:
            Sw+=i
        M=np.mean(data,axis=0)
        Sb=np.zeros([data.shape[1],data.shape[1]])
        for i in range(1,6):
            x=np.array([np.mean(data[(label==i)],axis=0)-M])
            Sb+=(np.matmul(x.T,x)*np.sum(label==i))
        SwSB=np.matmul(np.linalg.pinv(Sw),Sb)
        evals,evecs=np.linalg.eig(SwSB)
        W=evecs[:,np.argsort(evals)[::-1]][:,:2]
        self.projection=np.matmul(data,W)

    def run(self):
        self.Visualization()

    def Visualization(self):
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        for i in range(1,6):
            data=self.projection[(self.label==i)]
            plt.plot(data[:,0],data[:,1],color=color[i-1],linestyle='none',marker='o')
        # plt.show()
        plt.savefig('LDA.png')
        plt.close('all')