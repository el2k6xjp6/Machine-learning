import matplotlib.pyplot as plt
import math

def DisplayGraph(x,y,weight,xnum,ynum):
    """
    x,y:data
    weight:計算後的function
    xnum,ynum:x,y軸的刻度數量
    """
    (xmin,xmax,xgap,ymin,ymax,ygap)=(math.floor(min(x)),math.ceil(max(x)),math.ceil((math.ceil(max(x))-math.floor(min(x)))/xnum),math.floor(min(y)),math.ceil(max(y)),math.ceil((math.ceil(max(y))-math.floor(min(y)))/ynum))
    plt.axis([ xmin,xmax,ymin,ymax])
    plt.xticks([ i for i in range(xmin-xgap,xmax+xgap+1,xgap)])
    plt.yticks([ i for i in range(ymin-ygap,ymax+ygap+1,ygap)])
    plt.plot(x,y,"bo")
    a=[i+(j*0.1) for i in range (xmin-xgap,xmax+xgap) for j in range(10)]
    b=[]
    for i in a:
        num=0.0
        for j in range(len(weight)):
            num+=(weight[j][0]*(i**j))
        b.append(num)
    plt.plot(a,b,'r')
    plt.show()

def ReadFile(filepath):
    x=[]
    y=[]
    with open(filepath) as test:
        for line in test:
            temp=line.strip().split(",")
            if ( temp[0]!='' and temp[1]!='' ):
                x.append(float(temp[0]))
                y.append(float(temp[1]))
    return ( x , y )