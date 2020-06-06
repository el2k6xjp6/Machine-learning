import Data
import Matrix
import random
import math

class EM:
    def __init__(self):
        (self.image,self.label,self.probability)=Data.ReadMNISTData()
        self.weight=[[0.1] for i in range(10)]
        # self.probability=[[random.random()] for _ in range(10)]
        (self.positive,self.negative)=([],[])
        self.cluster=[[] for _ in range(len(self.label))]
        self.clabel=[]

    def __Estep(self):
        (self.positive,self.negative)=([],[])
        P=Matrix.multiply(Matrix.transpose(self.weight),self.probability)[0][0]
        N=Matrix.multiply(Matrix.transpose(self.weight),Matrix.subtraction([[1] for _ in range(len(self.probability))],self.probability))[0][0]
        for i in range(len(self.weight)):
            self.positive.append(self.weight[i][0] * self.probability[i][0] / P)
            self.negative.append(self.weight[i][0] * (1-self.probability[i][0]) / N)

    def __Mstep(self,data):
        (countP,countN)=(data.count(1),data.count(0))
        for i in range(len(self.weight)):
            self.weight[i][0]=(countP*self.positive[i]+countN*self.negative[i])/(countP+countN)
            self.probability[i][0]=(countP*self.positive[i])/(countP*self.positive[i]+countN*self.negative[i])

    def __Clustering(self):
        self.cluster=[[] for _ in range(len(self.label))]
        for i in range(len(self.image)):
            likelihood=[]
            (countP,countN)=(self.image[i].count(1),self.image[i].count(0))
            for j in range(len(self.weight)):
                # likelihood.append((countN+countP)*math.log(self.weight[j][0])+countP*math.log(self.probability[j][0])+countN*math.log(1-self.probability[j][0]))
                likelihood.append( self.positive[j]*(math.log(self.weight[j][0])+countP*math.log(self.probability[j][0])+countN*math.log(1-self.probability[j][0])))
            # print(likelihood.index(max(likelihood)))
            self.cluster[likelihood.index(max(likelihood))].append(i)
        for c in self.cluster:
            print(len(c))
        # a=[[] for i in range(10)]
        # counter=0
        # for c in self.cluster:
        #     count=[]
        #     for l in self.label:
        #         count.append(len(set(c).intersection(set(l))))
        #         a[counter].append(len(set(c).intersection(set(l))))
        #     # while (max(count) in self.clabel):
        #     #     print("counter = {}".format(counter))
        #     #     count.remove(max(count))
        #     self.clabel.append(max(count))
        #     counter+=1
        # print("="*30)
        # for i in a:
        #     print(i)
    # def __Clustering(self):
    #     for l in self.label:
    #         prediction=[]
    #         for i in l:
    #             data=self.image[i]
    #             (countP,countN)=(data.count(1),data.count(0))
    #             likelihood=[[0] for i in range(10)]
    #             for j in range(len(likelihood)):
    #                 likelihood[j]=(countN+countP)*math.log(self.weight[j][0])+countP*math.log(self.probability[j][0])+countN*math.log(1-self.probability[j][0])
    #             prediction.append(max(likelihood))
    #         count=[prediction.count(i) for i in range(10)]
    #         self.cluster.append(max(count))

    def __PrintResult(self):
        print("="*30)
        print(self.weight)
        print(self.probability)

    def run(self):
        for s in range(10):
            for i in self.image:
                self.__Estep()
                self.__Mstep(i)
            print("{}".format(s)+"="*30)
            print(self.weight)
            print(self.probability)
            print("{}".format(s)+"="*30)
            self.__Clustering()
        # self.__PrintResult()