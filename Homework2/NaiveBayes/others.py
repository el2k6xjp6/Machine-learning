import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def PrintImage(self,image):
    a = image.reshape(28,28)
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(a)
    plt.show()

def ReadTrainingData():
    with open('train-images-idx3-ubyte' , 'rb') as i:
        i.read(16)
        train_image = np.fromfile(i, dtype=np.uint8).reshape(60000,-1)
    with open('t10k-images-idx3-ubyte' , 'rb') as i:
        i.read(16)
        test_image = np.fromfile(i, dtype=np.uint8).reshape(10000,-1)
    with open('train-labels-idx1-ubyte', 'rb') as l:
        l.read(8)
        train_label = np.fromfile(l, dtype=np.int8)
    with open('t10k-labels-idx1-ubyte', 'rb') as l:
        l.read(8)
        test_label = np.fromfile(l, dtype=np.int8)

    return (train_image,train_label,test_image,test_label)
