import libsvm.svm as svm
import libsvm.svmutil as svmutil
import csv
import numpy as np
from math import exp
from scipy.spatial.distance import euclidean

class SVM:
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=self.ReadXCSV(X_train)
        self.Y_train=self.ReadYCSV(Y_train)
        self.X_test=self.ReadXCSV(X_test)
        self.Y_test=self.ReadYCSV(Y_test)

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
                accuracy=self.Training('-t 2 -c {} -g {} -v {}'.format(c,g,4),self.Y_train,self.X_train)
                if accuracy>best[2]:
                    best=(c,g,accuracy)
        print(best) # (1000, 0.01, 98.28)
        return (best[0],best[1])

    def UserDefineKernel(self,data1,data2,gamma):
        X=[[data1[i][j] for j in range(784)] for i in range(len(data1))]
        Y=[[data2[i][j] for j in range(784)] for i in range(len(data2))]
        kernel=[[i+1] for i in range(len(X))]
        for i in range(len(X)):
            for j in range(len(Y)):
                kernel[i].append(np.dot(X[i],Y[j])+exp((euclidean(X[i],Y[j])** 2)*(-1)*(gamma)))
        return kernel

    def run(self,mode):
        '''
        0-Linear Kernel 1-Polynomial Kernel 2-RBF Kernel 3-Best parameter RBF kernel 4-User define kernel
        '''
        if mode<=2:
            model=self.Training('-t {} -b 1'.format(mode),self.Y_train,self.X_train)
            svmutil.svm_predict(self.Y_test, self.X_test, model)
        elif mode==3:
            (C,gamma)=self.GridSearch()
            model=self.Training('-t 2 -c {} -g {}'.format(C,gamma),self.Y_train,self.X_train)
            svmutil.svm_predict(self.Y_test, self.X_test, model)
        elif mode==4:
            data=self.UserDefineKernel(self.X_train,self.X_train,0.01)
            print('{} {}'.format(len(data),len(self.Y_train)))
            model=self.Training('-t 4 -c {} -g {} -b 1'.format(1000,0.01),self.Y_train,data,True)
            test=self.UserDefineKernel(self.X_test,self.X_train,0.01)
            svmutil.svm_predict(self.Y_test,test,model)
            '''
            Model supports probability estimates, but disabled in predicton.
            Accuracy = 95.32% (2383/2500) (classification)
            '''