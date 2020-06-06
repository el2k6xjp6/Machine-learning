import random
import matplotlib.pyplot as plt

def UnivariateGaussianDataGenerator(mean,variance):
# The algorithms listed below all generate the standard normal deviates, 
# since a N(μ, σ^2) can be generated as X = μ + σZ, where Z is standard normal.
# An easy to program approximate approach, that relies on the central limit theorem, is as follows: 
# generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 
# the resulting random variable will have approximately standard normal distribution. 
# In truth, the distribution will be Irwin–Hall, which is a 12-section eleventh-order polynomial approximation to the normal distribution.
# This random deviate will have a limited range of (−6, 6).
    deviates = -6
    for _ in range(12):
        deviates += random.uniform(0, 1)
    return mean + deviates * (variance**0.5)

def  LinearModelDataGenerator(basis,weight,variance):
    x=random.uniform(-1,1)
    y=UnivariateGaussianDataGenerator(0,variance)
    for i in range(basis):
        y += (weight[i]*(x**i))
    return( x , y )

if __name__ == "__main__":
    X=[]
    Y=[]
    w=[1,2,1]
    for i in range(10000):
        (x,y)=LinearModelDataGenerator(3,w,20)
        X.append(x)
        Y.append(y)
    plt.plot(X,Y,"bo")
    plt.show()