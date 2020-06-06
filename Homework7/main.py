import numpy as np
import matplotlib.pyplot as plt
from PCA import PrincipalComponentsAnalysis
from LDA import LinearDiscriminantAnalysis
from SNE import SNE

label=np.loadtxt("data/mnist_label.csv",dtype=np.int)
data=np.loadtxt("data/mnist_X.csv",delimiter=",",dtype=np.float)

pca=PrincipalComponentsAnalysis(data,label)
pca.run()
# lda=LinearDiscriminantAnalysis(data,label)
# lda.run()
# sne=SNE(data,label)
# sne.run('t')
# sne.run('s')