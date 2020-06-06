from KMeans import KMeans
from KernelKMeans import KernelKMeans
from SpectralClustering import SpectralClustering
from DBSCAN import DBSCAN
import numpy as np

def ReadFile(file):
    data=[]
    with open(file,'r') as reader:
        for point in reader:
            p=point.strip().split(',')
            if len(p)==2:
                data.append([float(p[0]),float(p[1])])
    return np.array(data)

circle=ReadFile('circle.txt')
moon=ReadFile('moon.txt')

while 1:
    s=input("1 : K-Means | 2 : Kernel K-Means | 3 : Spectral clustering | 4 : DBSCAN\n")
    if s=="1":
        t=input("1 : circle | 2 : moon\n")
        if t=="1":
            KC=KMeans("circle",circle,2)
            KC.run()
            KC=None
        elif t=="2":
            KM=KMeans("moon",moon,2)
            KM.run()
            KM=None
    elif s=="2":
        t=input("1 : circle | 2 : moon\n")
        if t=="1":
            i=input("1 : original | 2 : initial\n")
            KKC=KernelKMeans('circle',circle,2,10,i=="2")
            KKC.run()
            KKC=None
        if t=="2":
            i=input("1 : original | 2 : initial\n")
            KKM=KernelKMeans('moon',moon,2,10,i=="2")
            KKM.run()
            KKM=None
    elif s=="3":
        t=input("1 : circle | 2 : moon\n")
        if t=="1":
            SCC=SpectralClustering('circle',circle,2,100)
            SCC.run()
            SCC=None
        if t=="2":
            SCM=SpectralClustering('moon',moon,2,100)
            SCM.run()
            SCM=None
    if s=="4":
        t=input("1 : circle | 2 : moon\n")
        if t=="1":
            DC=DBSCAN('circle',circle,0.1,3)
            DC.run()
            DC=None
        if t=="2":
            DM=DBSCAN('moon',moon,0.2,5)
            DM.run()
            DM=None