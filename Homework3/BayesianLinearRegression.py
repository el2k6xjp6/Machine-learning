import Data
import Matrix
import matplotlib.pyplot as mp

class Regression:
    def __init__(self,b,n,a,w):
        self.covariance=Matrix.diagonal(n,b) #posterior covariance matrix
        self.mean=[[0] for i in range(n)] #posterior mean matrix
        self.basis=n
        self.a=a #noise covariance
        self.weight=w
        self.counter=0 #quantity of incomes
        self.mten=[] #mean of ten incomes
        self.vten=0 #variance of ten incomes
        self.mfifty=[] #mean of fifty incomes
        self.vfifty=0 #variance of fifty incomes
        self.x=0
        self.y=0
        self.X=[] # x incomes
        self.Y=[] # y incomes

    def __Converge(self,prior,posterior):
        sum=0
        for i in range(len(prior)):
            for j in range(len(prior[0])):
                sum += (prior[i][j]-posterior[i][j])
        return abs(sum)<0.0000001

    def __OnlineLearning(self,mean,covariance,X):
#      Posterior:
#      Mean=a(X^T)*X+bI //  a(X^T)*X+(prior covariance)
        self.covariance=Matrix.addition(Matrix.cmultiply(Matrix.multiply(Matrix.transpose(X),X),self.a),covariance)
#      Covariance matrix=a(covariance^-1)(X^T)y + (covariance^(-1))(prior covariance)(prior mean)
        self.mean=Matrix.addition(Matrix.cmultiply(Matrix.multiply(Matrix.getMatrixInverse(self.covariance),Matrix.transpose(X)),self.a*self.y),Matrix.multiply(Matrix.multiply(Matrix.getMatrixInverse(self.covariance),covariance),mean))
        if self.counter==10:
            self.mten=self.mean
            self.vten=self.covariance
        if self.counter==50:
            self.mfifty=self.mean
            self.vfifty=self.covariance
        print("\nPostirior mean:")
        for m in self.mean:
            print(m[0])
        print("\nPostirior variance:")
        for c in Matrix.getMatrixInverse(self.covariance):
            row=""
            for i in range(len(c)):
                if i>0:
                    row+=" , "
                row+=str(round(c[i],10))
            print(row)
#      Predictive distribution
#      predictive mean=(mean^T)*X
#      predictive variance=1/a + (X^T)*(posterior covariance matrix)*X
        print("\nPredictive distribution ~ N( {} , {} )".format(Matrix.multiply(Matrix.transpose(mean),X)[0][0],(1/self.a)+Matrix.multiply(Matrix.multiply(X,Matrix.getMatrixInverse(covariance)),Matrix.transpose(X))[0][0]))
        return (self.__Converge(self.mean,mean)) & (self.__Converge(Matrix.getMatrixInverse(self.covariance),Matrix.getMatrixInverse(covariance)))

    def __DisplayGraph(self,type,weight,covariance):
        A=[i+(j*0.1) for i in range (-2,2) for j in range(10)]
        B=[]
        Up=[]
        Down=[]
        for i in A:
            num=0.0
            for j in range(len(weight)):
                num+=(weight[j]*(i**j))
            B.append(num)
            X=[[i ** j for j in range(self.basis)]]
            if type==0:
                variance=covariance
            else:
                variance=((1/self.a)+Matrix.multiply(Matrix.multiply(X,Matrix.getMatrixInverse(covariance)),Matrix.transpose(X))[0][0]) ** (1/2)
            Up.append(num+variance)
            Down.append(num-variance)
        mp.axis([-2,2,-20,25])
        mp.xticks([ i for i in range(-2,3,1)])
        mp.yticks([ i for i in range(-20,21,10)])
        if type==1:
            mp.plot(self.X[:10],self.Y[:10],'bo')
        elif type==2:
            mp.plot(self.X[:50],self.Y[:50],'bo')
        elif type==3:
            mp.plot(self.X,self.Y,'bo')
        mp.plot(A,B,'k')
        mp.plot(A,Up,'r',antialiased=True)
        mp.plot(A,Down,'r',antialiased=True)
        mp.show()

    def __Visualization(self):
        self.__DisplayGraph(0,self.weight,1/self.a)
        self.__DisplayGraph(1,[i[0] for i in self.mten],self.vten)
        self.__DisplayGraph(2,[i[0] for i in self.mfifty],self.vfifty)
        self.__DisplayGraph(3,[i[0] for i in self.mean],self.covariance)

    def Run(self):
        while(1):
            (self.x,self.y)=Data.LinearModelDataGenerator(self.basis,self.weight,self.a)
            self.X.append(self.x)
            self.Y.append(self.y)
            print("Add data point ( {} , {} ):".format(self.x,self.y))
            self.counter+=1
            if self.__OnlineLearning(self.mean,self.covariance,[[self.x ** i for i in range(self.basis)]]):
                break
            print("="*25)
        self.__Visualization()