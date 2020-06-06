import math
import numpy as np
import pandas as pd
import os.path

class NaiveBayes:
    def __init__(self,train_image,train_label,test_image,test_label,bins):
        self.train_image=train_image
        self.train_label=train_label
        self.test_image=test_image
        self.test_label=test_label
        self.bins=bins

    def __TallyPixel(self):
        self.train_image = self.train_image // (256/self.bins)
        self.test_image = self.test_image // (256/self.bins)

    def __PriorProbability(self):
        prior = np.zeros(10)
        print("Prior:")
        for i in range(10):
            total = len(self.train_label)
            prior[i] =math.log( np.count_nonzero(self.train_label == i) / total)
            print("{} : {}".format(i,math.exp(prior[i])))
        return prior

    def __Likelihood(self):
        train=pd.DataFrame(self.train_image)
        train['Label']=self.train_label
        likelihood = [[[0 for i in range(self.bins)] for j in range(len(self.train_image))] for k in range(10)]
        if os.path.isfile('likelihood.npy'):
            likelihood=np.load('likelihood.npy')
        else:
            for i in range(10):
                for j in range(784):
                    total=len(train[train['Label']==i])
                    for k in range(self.bins):
                        count=len(train[(train[j]==k) & (train['Label']==i)])
                        if count==0:
                            count+=1
                        likelihood[i][j][k]=math.log(count/total)
                        print('{} {} {} {}'.format(i,j,k,likelihood[i][j][k]))
            np.save('likelihood.npy',likelihood)
        return likelihood

    def __Predict(self,prior,likelihood):
        prediction=[i for i in self.test_label]
        for i in range(len(self.test_image)):
            print('='*25)
            print('No. {}'.format(i))
            posterior=[i for i in prior]
            for j in range(len(prior)):
                for k in range(len(self.test_image[0])):
                    posterior[j]+=likelihood[j][k][int(self.test_image[i][k])]
            _sum=0
            for n in posterior:
                _sum+=n
            prediction[i]-=np.argmax(posterior)
            print('Posterior:')
            for n in range(10):
                print('{}: {}'.format(n,posterior[n]/_sum))
            print('Prediction: {}, Answer: {}'.format(np.argmax(posterior),self.test_label[i]))
        self.__PrintDiscreteImagination(likelihood)
        print('\nError rate: {}'.format((np.count_nonzero(prediction)/len(self.test_label))))

    def __PrintDiscreteImagination(self,likelihood):
        for r in range(10):
            image=[[0 for i in range(28)] for j in range(28)]
            for i in range(28):
                for j in range(28):
                    a=[0,0]
                    for k in range(16):
                        a[0]+=math.exp(likelihood[r][28*i+j][k])
                        a[1]+=math.exp(likelihood[r][28*i+j][k+16])
                    if a[0] < a[1]:
                        image[i][j]=1
            print('Imagination of {}:'.format(r))
            for i in image:
                print(i)

    def __CalculateMeanAndStandardDeviation(self):
        train=pd.DataFrame(self.train_image)
        train['Label']=self.train_label
        mean=[]
        std=[]
        for i in range(10):
            target = train[train['Label'] == i].drop(['Label'], axis=1).values
            mean.append([i for i in np.mean(target, axis=0)])
            std.append([i for i in np.std(target, axis=0)])
        print('{} {} '.format(len(mean),len(mean[0])))
        return(mean,std)

    def __Gaussian(self,x, mean, std):
        if std <60:
            std = 60
        answer = math.log(1 / (std * math.sqrt(2 * math.pi))) - (0.5 * (  math.pow(  (x-mean) / std   ,  2 ) ) )
        return answer

    def __GaussianDistribution(self,prior,mean,std):
        prediction=[i for i in self.test_label]
        for i in range(len(self.test_image)):
            print('='*25)
            print('No. {}'.format(i))
            posterior=[i for i in prior]
            for j in range(10):
                for k in range(len(self.test_image[0])):
                    x = self.test_image[i][k]
                    Mean = mean[j][k]
                    Std = std[j][k]
                    posterior[j] += self.__Gaussian(x,Mean,Std)
            _sum=0
            for n in posterior:
                _sum+=n
            prediction[i]-=np.argmax(posterior)
            print('Posterior:')
            for n in range(10):
                print('{}: {}'.format(n,posterior[n]/_sum))
            print('Prediction: {}, Answer: {}'.format(np.argmax(posterior),self.test_label[i]))
        self.__PrintContinuousImagination(mean,std)
        print('\nError rate: {}'.format((np.count_nonzero(prediction)/len(self.test_label))))

    def __PrintContinuousImagination(self,mean,std):
        for n in range(10):
            image=[[0 for i in range(28)] for j in range(28)]
            for i in range(28):
                for j in range(28):
                    a=[0,0]
                    for k in range(128):
                        a[0]+=self.__Gaussian(k,mean[n][28*i+j],std[n][28*i+j])
                        a[1]+=self.__Gaussian(k+128,mean[n][28*i+j],std[n][28*i+j])
                    if a[0] < a[1]:
                        image[i][j]=1
            print('Imagination of {}:'.format(n))
            for i in image:
                print(i)

    def __Discrete(self):
        self.__TallyPixel()
        prior=self.__PriorProbability()
        likelihood=self.__Likelihood()
        self.__Predict(prior,likelihood)
        

    def __Continuous(self):
        prior=self.__PriorProbability()
        (mean,std)=self.__CalculateMeanAndStandardDeviation()
        self.__GaussianDistribution(prior,mean,std)
        

    def run(self,mode):
        if mode :
            self.__Continuous()
        else:
            self.__Discrete()