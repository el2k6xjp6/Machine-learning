import matrix as matrix
import others as others

class LSE:
    def __init__(self, x , y , degree , lm):
        self.x=x
        self.y=[[i] for i in y]
        self.degree=degree
        self.lm=lm
    def CreateFeatures(self):
        features=[]
        for data in self.x:
            temp=[]
            for power in range(self.degree):
                temp.append(data**power)
            features.append(temp)
        return features

    def CreateMatrix(self,features):
        AtA=matrix.multiply(matrix.transpose(features),features)
        return matrix.addition( AtA , matrix.diagonal(self.degree,self.lm) )

    def CalculateError(self,weight):
        error=0.0
        weight=matrix.transpose(weight)
        for (xi,yi) in zip(self.x,self.y):
            predict=0.0
            for d in range(self.degree):
                predict+=(weight[0][d] * (xi**d))
            error+=(predict-yi[0])**2
        return error

    def PrintResult(self,weight,error):
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
        others.DisplayGraph(self.x,[yi[0] for yi in self.y],weight,7,7)

    def run(self):
        features=self.CreateFeatures()
        M=self.CreateMatrix(features)
        b=matrix.multiply(matrix.transpose(features),self.y)
        A=matrix.inverse(M)
        weight=matrix.multiply(A,b)
        error=self.CalculateError(weight)
        self.PrintResult(weight,error)
