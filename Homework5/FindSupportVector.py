import csv
import libsvm.svm as svm
import libsvm.svmutil as svmutil
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from scipy.spatial.distance import euclidean

class SV:
    def __init__(self,Plot_X,Plot_Y):
            self.Plot_X=self.ReadXCSV(Plot_X)
            self.Plot_Y=self.ReadYCSV(Plot_Y)

    def ReadXCSV(self,filename):
        X=[]
        with open(filename) as f:
            images=csv.reader(f)
            for image in images:
                temp={}
                for pixel in range(len(image)):
                    temp[pixel]=float(image[pixel])
                X.append(temp)
        return X

    def ReadYCSV(self,filename):
        Y=[]
        with open(filename) as f:
            labels=csv.reader(f)
            for label in labels:
                Y.append(int(label[0]))
        return Y

    def Training(self,kernel,target,data,IsKernel=False):
        problem=svm.svm_problem(target,data,isKernel=IsKernel)
        parameter=svm.svm_parameter(kernel)
        model=svmutil.svm_train(problem,parameter)
        return model

    def GridSearch(self):
        C=[10 **d for d in range(-4,5)]
        Gamma=[10 **d for d in range(-4,5)]
        best=(0,0,0) #[C,gamma,accuracy]
        for c in C:
            for g in Gamma:
                accuracy=self.Training('-t 2 -c {} -g {} -v {}'.format(c,g,10),self.Plot_Y,self.Plot_X)
                if accuracy>best[2]:
                    best=(c,g,accuracy)
        print(best)
        return (best[0],best[1])

    def UserDefineKernel(self,data1,data2,gamma):
        X=[[data1[i][j] for j in range(2)] for i in range(3000)]
        Y=[[data2[i][j] for j in range(2)] for i in range(3000)]
        kernel=[[i+1] for i in range(len(X))]
        for i in range(len(X)):
            for j in range(len(Y)):
                kernel[i].append(np.dot(X[i],Y[j])+exp((euclidean(X[i],Y[j])** 2)*(-1)*(gamma)))
        return kernel

    def Visualization(self,SV,label):
        (C0x,C0y,C1x,C1y,C2x,C2y)=([],[],[],[],[],[])
        (V0x,V0y,V1x,V1y,V2x,V2y)=([],[],[],[],[],[])
        for i in range(len(self.Plot_X)):
            if int(label[i]) ==0:
                if i in SV:
                    V0x.append(self.Plot_X[i][0])
                    V0y.append(self.Plot_X[i][1])
                else:
                    C0x.append(self.Plot_X[i][0])
                    C0y.append(self.Plot_X[i][1])
            elif int(label[i]) ==1:
                if i in SV:
                    V1x.append(self.Plot_X[i][0])
                    V1y.append(self.Plot_X[i][1])
                else:
                    C1x.append(self.Plot_X[i][0])
                    C1y.append(self.Plot_X[i][1])
            elif int(label[i]) ==2:
                if i in SV:
                    V2x.append(self.Plot_X[i][0])
                    V2y.append(self.Plot_X[i][1])
                else:
                    C2x.append(self.Plot_X[i][0])
                    C2y.append(self.Plot_X[i][1])
        plt.plot(C0x,C0y,'ro')
        plt.plot(C1x,C1y,'go')
        plt.plot(C2x,C2y,'bo')
        plt.plot(V0x,V0y,'rx')
        plt.plot(V1x,V1y,'gx')
        plt.plot(V2x,V2y,'bx')
        plt.show()

    def run(self,mode):
        if mode<=2:
            model=self.Training('-t {} -b 1'.format(mode),self.Plot_Y,self.Plot_X)
            sv=model.get_sv_indices()
            (label,_,_)=svmutil.svm_predict(self.Plot_Y,self.Plot_X, model)
            svmutil.svm_predict(self.Y_test, self.X_test, model)
            self.Visualization([i-1 for i in sv],label)
        # elif mode==3:
        #     (C,gamma)=self.GridSearch()
        #     model=self.Training('-t 2 -c {} -g {}'.format(C,gamma),self.Plot_Y,self.Plot_X)
        #     sv=model.get_sv_indices()
        #     (label,_,_)=svmutil.svm_predict(self.Plot_Y,self.Plot_X,model)
        #     self.Visualization([i-1 for i in sv],label)
        elif mode==3:
            data=self.UserDefineKernel(self.Plot_X,self.Plot_X,0.1)
            model=self.Training('-t 4 -c {} -g {} -b 1'.format(0.01,0.1),self.Plot_Y,data,True)
            sv=model.get_sv_indices()
            (label,_,_)=svmutil.svm_predict(self.Plot_Y,data,model)
            self.Visualization([i-1 for i in sv],label)