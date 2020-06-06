import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as ml

class PrincipalComponentsAnalysis():
    def __init__(self,data,label):
        self.data=data
        self.label=label

    def run(self):
        mean=ml.repmat(np.mean(self.data,axis=0),self.data.shape[0],1)
        X=self.data-mean
        print(X.shape)
        U,_,_=np.linalg.svd(X.T)
        print(U.shape)
        self.projection=np.dot(self.data,U[:,:2])
        # self.Visualization()

    def Visualization(self):
        color=['#87CEFA','#FFB7DD','#ca8eff','#ff9d6f','#ff7575']
        for i in range(1,6):
            data=self.projection[(self.label==i)]
            plt.plot(data[:,0],data[:,1],color=color[i-1],linestyle='none',marker='o')
        # plt.show()
        plt.savefig('PCA.png')
        plt.close('all')
