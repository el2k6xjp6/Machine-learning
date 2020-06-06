import Data
import math
import Matrix
import matplotlib.pyplot as plt

class Logistic:
    def __init__(self,num,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2):
        self.num=num
        self.mx1=mx1
        self.vx1=vx1
        self.my1=my1
        self.vy1=vy1
        self.mx2=mx2
        self.vx2=vx2
        self.my2=my2
        self.vy2=vy2
        self.data=[]
        self.target=[]
        self.Gweight=[]
        self.Nweight=[]
        self.Gprediction=[]
        self.Nprediction=[]
        self.learningrate=0.01

    def __CreateData(self):
        for _ in range(self.num):
            x = Data.UnivariateGaussianDataGenerator(self.mx1,self.vx1)
            y = Data.UnivariateGaussianDataGenerator(self.my1,self.vy1)
            self.data.append([1,x, y])
            self.target.append([0])
        for _ in range(self.num):
            x = Data.UnivariateGaussianDataGenerator(self.mx2,self.vx2)
            y = Data.UnivariateGaussianDataGenerator(self.my2,self.vy2)
            self.data.append([1,x, y])
            self.target.append([1])

    def __Sigmoid(self,z):
        return 1/(1+math.exp(-z))

    def __Converge(self,gradient):
        Sum=0
        for g in gradient:
            Sum+=(abs(g[0]))
        return (Sum<0.01)

    def __Gradient(self,weight):
        Prediction=[[self.__Sigmoid(Matrix.multiply([x],weight)[0][0])] for x in self.data]
        return Matrix.multiply(Matrix.transpose(self.data),Matrix.subtraction(Prediction,self.target))
  
    def __Hessian(self):
        D=[[0 for i in range(len(self.data))] for j in range(len(self.data))]
        for i in range(len(self.data)):
            D[i][i]=(self.__Sigmoid(Matrix.multiply([self.data[i]],self.Nweight)[0][0]))*(1-(self.__Sigmoid(Matrix.multiply([self.data[i]],self.Nweight)[0][0])))
        H=Matrix.multiply(Matrix.transpose(self.data),Matrix.multiply(D,self.data))
        return Matrix.Inverse(H)

    def __GradientDecent(self):
        self.Gweight=[[0] for i in range(len(self.data[0]))]
        count=0
        while(1):
            count+=1
            print(count)
            gradient=self.__Gradient(self.Gweight)
            print(gradient)
            self.Gweight=Matrix.subtraction(self.Gweight,Matrix.cmultiply(gradient,self.learningrate))
            if self.__Converge(gradient):
                prediction=[self.__Sigmoid(Matrix.multiply([x],self.Gweight)[0][0]) for x in self.data]
                for i in range(len(prediction)):
                    if prediction[i]<0.5:
                        self.Gprediction.append(0)
                    else:
                        self.Gprediction.append(1)
                break

    def __NewtonMethod(self):
        self.Nweight=[[0] for i in range(len(self.data[0]))]
        count=0
        while(1):
            count+=1
            print(count)
            gradient=self.__Gradient(self.Nweight)
            hessian=self.__Hessian()
            # update=Matrix.multiply(hessian,Matrix.cmultiply(gradient,self.learningrate))
            if Matrix.Deternminant(hessian)==0:
                update=Matrix.cmultiply(gradient,self.learningrate)
            else:
                update=Matrix.multiply(hessian,Matrix.cmultiply(gradient,self.learningrate))
            print(update)
            self.Nweight=Matrix.subtraction(self.Nweight,update)
            if self.__Converge(update):
                prediction=[self.__Sigmoid(Matrix.multiply([x],self.Nweight)[0][0]) for x in self.data]
                for i in range(len(prediction)):
                    if prediction[i]<0.5:
                        self.Nprediction.append(0)
                    else:
                        self.Nprediction.append(1)
                break

    def __Classify(self,prediction):
        (x1,y1,x2,y2)=([],[],[],[])
        for i in range(len(prediction)):
            if prediction[i]==0:
                x1.append(self.data[i][1])
                y1.append(self.data[i][2])
            else:
                x2.append(self.data[i][1])
                y2.append(self.data[i][2])
        return (x1,y1,x2,y2)

    def __PrintGraph(self,n):
        if n==0:
            (x1,y1,x2,y2)=self.__Classify([t[0] for t in self.target])
            plt.subplot(131)
            plt.title("Ground truth")
        elif n==1:
            (x1,y1,x2,y2)=self.__Classify(self.Gprediction)
            plt.subplot(132)
            plt.title("Gradient decent")
        elif n==2:
            (x1,y1,x2,y2)=self.__Classify(self.Nprediction)
            plt.subplot(133)
            plt.title("Newton's method")
        plt.plot(x1,y1,'ro')
        plt.plot(x2,y2,'bo')

    def __PrintConfusionMatrix(self,n):
        (TP,FP,FN,TN)=(0,0,0,0)
        if n:
            print("Newton's method:")
            prediction=self.Nprediction
            weight=self.Nweight
        else:
            print("Gradient decent:")
            prediction=self.Gprediction
            weight=self.Gweight
        for i in range(len(self.target)):
            if (self.target[i][0]==0) and (prediction[i]==0):
                TP+=1
            elif (self.target[i][0]==1) and (prediction[i]==0):
                FP+=1
            elif (self.target[i][0]==0) and (prediction[i]==1):
                FN+=1
            elif (self.target[i][0]==1) and (prediction[i]==1):
                TN+=1
        print("="*60)
        print("w:")
        for i in weight:
            print(i)
        print("\nConfusion Matrix:")
        print("              Predict cluster 1 Predict cluster 2")
        print("Is cluster 1         {}                  {}       ".format(TP,FN))
        print("Is cluster 2         {}                  {}       ".format(FP,TN))
        print("Sensitivity (Successfully predict cluster 1) : {}".format(TP/(TP+FN)))
        print("Specificity (Successfully predict cluster 2) : {}".format(TN/(TN+FP)))

    def __PrintResult(self):
        for i in range(3):
            self.__PrintGraph(i)
        plt.show()
        for i in range(2):
            self.__PrintConfusionMatrix(i)

    def run(self):
        self.__CreateData()
        self.__GradientDecent()
        self.__NewtonMethod()
        self.__PrintResult()
