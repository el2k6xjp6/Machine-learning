import Data

class Estimator:
    def __init__(self,mean,variance):
        self.mean=mean
        self.variance=variance
        self.estimatemean=Data.UnivariateGaussianDataGenerator(mean,variance)
        self.estimatevariance=0
        self.M=0
        self.point=0
        self.count=1

    def __WelfordsOnlineAlgorithm(self,mean,M,variance):
#       Mean(n) = [(n-1)Mean(n-1)+Xn]/n = Mean(n-1)+[Xn-Mean(n-1)]/n
#       M(2,n) = M(2,n-1)+(Xn-Mean(n-1))*(Xn-Mean(Xn))
#       Var(n)=M(2,n)/(n-1)
        self.count+=1
        self.estimatemean=mean+(self.point-mean)/self.count
        self.M=M+(self.point-mean)*(self.point-self.estimatemean)
        self.estimatevariance=self.M/(self.count-1)
        print('Estimate Mean: {}'.format(self.estimatemean))
        print('Estimate variance: {}'.format(self.estimatevariance))
        return (abs(self.estimatemean-mean)<0.00001) & (abs(self.variance-variance)<0.00001) 

    def Estimate(self):
        while(1):
            print('='*20)
            self.point=Data.UnivariateGaussianDataGenerator(self.mean,self.variance)
            print('Current Point: {}'.format(self.point))
            if self.__WelfordsOnlineAlgorithm(self.estimatemean,self.M,self.estimatevariance):
                break
