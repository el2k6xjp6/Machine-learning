import OnlineLearning

count=0
filepath=input("Filepath: ")
while(1):
    count+=1
    a=input("a:")
    b=input("b:")
    print("Case {}: a={}, b={}".format(count,a,b))
    OL=OnlineLearning.OnlineLearning(filepath,int(a),int(b))
    OL.run()