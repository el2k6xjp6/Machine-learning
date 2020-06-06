import LSE as LSE
import Newton as Newton
import others as others

case=0
while(1):
    case+=1
    # filepath=input("File path:")
    filepath="testfile.txt"
    degree=int(input("n:"))
    lm=input("Lambda:")
    print("Case {}: n={}, lambda={}".format(case,degree,lm))
    (x,y)=others.ReadFile(filepath)
    print("LSE:")
    lse=LSE.LSE(x,y,degree,lm)
    lse.run()
    print("Newton' Method:")
    newton=Newton.Newton(x,y,degree)
    newton.run()