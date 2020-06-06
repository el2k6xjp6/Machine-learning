def dimension(A):
    return (len(A),len(A[0]))

def diagonal(n,x):
    A=[ [0 for row in range(n) ] for col in range(n) ]
    for i in range (n):
        A[i][i]=x
    return A

def cmultiply (A,c):
    C=[ [0 for row in range(len(A[0]))] for col in range(len(A)) ]
    for m in range(len(A)):
        for n in range(len(A[0])):
            C[m][n]=A[m][n] * c
    return C

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

def matrixcopy(A):
    C=[[0 for row in range(len(A))] for col in range(len(A[0]))]
    for m in range(len(A[0])):
        for n in range(len(A)):
            C[m][n]=A[m][n]
    return C

# def LU(A):
#     U=matrixcopy(A)
#     (m,n)=dimension(A)
#     L=diagonal(m,1)
#     for i in range(m-1):
#         for k in range(i+1,m):
#             t=-U[k][i]/U[i][i]
#             for j in range(i,n):
#                 U[k][j]=U[k][j]+(t*U[i][j])
#             L[k][i]=-t
#     return(L,U)

# def inverse(A):
#     (L,U)=LU(A)
#     n=len(A)
#     X=[ [0 for row in range(n)] for col in range(n)]
#     for i in range(n): 
#         X[i][i]=1/L[i][i]
#         for j in range(i+1):
#             L[i][j]=L[i][j]/L[i][i]
#     for i in range(n):
#         for k in range(i+1,n):
#             t=-L[k][i]/L[i][i]
#             for j in range(i+1):
#                 L[k][j]=L[k][j]+t*L[i][j]
#                 X[k][j]=X[k][j]+t*X[i][j]
#     Y=[ [0 for row in range(n)] for col in range(n)]
#     for i in range(n):
#         Y[i][i]=1/U[i][i]
#         t=U[i][i]
#         for j in range(i,n):
#             U[i][j]=U[i][j]/t
#     for i in range(n-1,-1,-1):
#         for k in range(i-1,-1,-1):
#             t=-U[k][i]/U[i][i]
#             for j in range(n-1,i-1,-1):
#                 U[k][j]=U[k][j]+t*U[i][j]
#                 Y[k][j]=Y[k][j]+t*Y[i][j]
#     return multiply(Y,X)

def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(A):
    m=matrixcopy(A)
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors