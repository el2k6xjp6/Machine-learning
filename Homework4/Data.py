from matplotlib import pyplot as plt
import numpy as np
import random

# def PrintImage(self,image):
#     a = image.reshape(28,28)
#     fig=plt.gcf()
#     fig.set_size_inches(2,2)
#     plt.imshow(a)
#     plt.show()

def ReadMNISTData():
    with open('train-images-idx3-ubyte' , 'rb') as i:
        i.read(16)
        train_image = np.fromfile(i, dtype=np.uint8).reshape(60000,-1)
    with open('train-labels-idx1-ubyte', 'rb') as l:
        l.read(8)
        train_label = np.fromfile(l, dtype=np.int8)
    image=train_image.tolist()
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]>127:
                image[i][j]=1
            else:
                image[i][j]=0
    label=[[] for _ in range(10)]
    for i in range(len(train_label)):
        label[train_label[i]].append(i)
    probability=[]
    for l in label:
        p=0
        for i in l:
            p+=image[i].count(1)
        probability.append([(p/len(l))/784])
    return (image,label,probability)

def UnivariateGaussianDataGenerator(mean,variance):
    deviates = -6
    for _ in range(12):
        deviates += random.uniform(0, 1)
    return mean + deviates * (variance**0.5)
