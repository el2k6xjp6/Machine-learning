import matrix as matrix
import others as others

class Newton:
    def __init__(self,x,y,degree):
        self.x=[[i] for i in x] 
        self.y=[[j] for j in y]
        self.degree=degree

    def __CreateFeatures(self):
        features=[]
        for data in self.x:
            temp=[]
            for power in range(self.degree):
                temp.append(data[0]**power)
            features.append(temp)
        return features

    def __HessionInverse(self,features):
        AtA=matrix.multiply(matrix.transpose(features),features)
        return matrix.inverse(AtA)

    def __Gradient(self,features,weight):
        AtAx=matrix.multiply(matrix.multiply(matrix.transpose(features),features),weight)
        Atb=matrix.multiply(matrix.transpose(features),self.y)
        matrix.addition(AtAx,Atb)
        return matrix.subtraction(AtAx,Atb)

    def __NewtonMethod(self,hesion,gradient,weight):
        return matrix.subtraction(weight,matrix.multiply(hesion,gradient))

    def __CalculateError(self,weight):
        error=0.0
        weight=matrix.transpose(weight)
        for (xi,yi) in zip(self.x,self.y):
            predict=0.0
            for d in range(self.degree):
                predict+=(weight[0][d] * (xi[0]**d))
            error+=(predict-yi[0])**2
        return error

    def __PrintResult(self,weight,error):
        w=""
        for i in range(self.degree-1,-1,-1):
            if i!=(self.degree-1):
                w+=" + "
            if (i==0):
                w+="{}".format(weight[i][0])
            else:
                w+="{}X^{}".format(weight[i][0],i)
        print("Fitting line: "+w)
        print("Total error: {}".format(error))
        others.DisplayGraph([xi[0] for xi in self.x],[yi[0] for yi in self.y],weight,7,7)

    def run(self):
        weight = [[0] for i in range(self.degree)]
        feature = self.__CreateFeatures()
        Hession=self.__HessionInverse(feature)
        gradient=self.__Gradient(feature,weight)
        weight=self.__NewtonMethod(Hession,gradient,weight)
        error=self.__CalculateError(weight)
        self.__PrintResult(weight,error)