import numpy as np
A = np.array([[1.0,2.0,3.0],[4.0,4.0,4.0],[2.0,2.0,1.0]])
print(A)
U=A.copy()
L=np.zeros((3,3))
for i in range(len(L)):
    L[i][i]=1

for col in range(len(U[0])):
    div = U[col][col]
    for row in range(col+1,len(U)):
        mult=U[row][col]/div
        for i in range(len(U[row])):
            U[row][i]-=U[col][i]*mult
        L[row][col]=mult

print(L)
print(U)
print(np.matmul(L,U))
print(np.subtract(np.matmul(L,U),A))