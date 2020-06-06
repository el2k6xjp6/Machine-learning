from SVMonMNIST import SVM 
from FindSupportVector import SV

part1=SVM('X_train.csv','Y_train.csv','X_test.csv','Y_test.csv')
part2=SV('Plot_X.csv','Plot_Y.csv')

while 1:
    part=int(input(" 1-Part 1 | 2-Part 2\n"))
    if part==1:
        mode=int(input(" 0-Linear Kernel | 1-Polynomial Kernel | 2-RBF Kernel | 3-Best parameter RBF kernel | 4-User define kernel | 5-All\n"))
        if mode==5:
            for i in range(5):
                part1.run(i)
        else:
            part1.run(mode)
    elif part==2:
        mode=int(input(" 0-Linear Kernel | 1-Polynomial Kernel | 2-RBF Kernel | 3-User define kernel | 4-All\n"))
        if mode ==4:
            for i in range(5):
                part2.run(i)
        else:
            part2.run(mode)