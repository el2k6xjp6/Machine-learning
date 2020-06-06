def dimension(A):
    return (len(A),len(A[0]))

def diagonal(n,x):
    A=[ [0 for row in range(n) ] for col in range(n) ]
    for i in range (n):
        A[i][i]=x
    return A

def multiply (A,B):
    C=[ [0 for row in range(len(B[0]))] for col in range(len(A)) ]
    for m in range(len(A)):
        for n in range(len(B[0])):
            for r in range(len(B)):
                C[m][n]+=A[m][r] * B[r][n]
    return C

def addition(A,B):
    C=[ [0 for row in range(len(A[0]))] for col in range(len(A)) ]
    for m in range(len(A)):
        for n in range(len(A[0])):
            C[m][n]=float(A[m][n])+float(B[m][n])
    return C

def subtraction(A,B):
    C=[ [0 for row in range(len(A[0]))] for col in range(len(A)) ]
    for m in range(len(A)):
        for n in range(len(A[0])):
            C[m][n]=float(A[m][n])-float(B[m][n])
    return C

def transpose(A):
    C=[ [0 for row in range(len(A))] for col in range(len(A[0])) ]
    for m in range(len(A[0])):
        for n in range(len(A)):
            C[m][n]=A[n][m]
    return C

def LU(A):
    U=A
    (m,n)=dimension(A)
    L=diagonal(m,1)
    for i in range(m-1):
        for k in range(i+1,m):
            t=-U[k][i]/U[i][i]
            for j in range(i,n):
                U[k][j]=U[k][j]+(t*U[i][j])
            L[k][i]=-t
    return(L,U)

def inverse(A):
    (L,U)=LU(A)
    n=len(A)
    X=[ [0 for row in range(n)] for col in range(n)]
    for i in range(n): 
        X[i][i]=1/L[i][i]
        for j in range(i+1):
            L[i][j]=L[i][j]/L[i][i]
    for i in range(n):
        for k in range(i+1,n):
            t=-L[k][i]/L[i][i]
            for j in range(i+1):
                L[k][j]=L[k][j]+t*L[i][j]
                X[k][j]=X[k][j]+t*X[i][j]
    Y=[ [0 for row in range(n)] for col in range(n)]
    for i in range(n):
        Y[i][i]=1/U[i][i]
        t=U[i][i]
        for j in range(i,n):
            U[i][j]=U[i][j]/t
    for i in range(n-1,-1,-1):
        for k in range(i-1,-1,-1):
            t=-U[k][i]/U[i][i]
            for j in range(n-1,i-1,-1):
                U[k][j]=U[k][j]+t*U[i][j]
                Y[k][j]=Y[k][j]+t*Y[i][j]
    return multiply(Y,X)
