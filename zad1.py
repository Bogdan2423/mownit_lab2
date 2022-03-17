import numpy as np
import scipy

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
B = np.array([3, 2, 1])


# A[[0,2]]=A[[2,0]]

def partial_pivot(A, col):
    pivot_val = abs(A[0][col])
    pivot_index = 0
    for j in range(col, len(A)):
        if abs(A[j][col]) > pivot_val:
            pivot_val = A[j][col]
            pivot_index = j
    return pivot_index


def swap_rows(A, B, i, j):
    A[[i, j]] = A[[j, i]]
    B[i], B[j] = B[j], B[i]


def make_pivot_one(A, B, i):
    pivot = A[i][i]
    for j in range(i, len(A[i])):
        A[i][j] = A[i][j]/pivot
    B[i] = B[i]/pivot

def subtract_rows(A,B,i):
    #for pivot==1
    for j in range(len(A)):
        if j!=i:
            temp=[el*A[j][i] for el in A[i]]
            A[j]=np.subtract(A[j],temp)
            B[j]-=B[i]*A[j][i]

for i in range(len(A[0])):
    pivot_index = partial_pivot(A, i)
    swap_rows(A, B, i, pivot_index)
    print(A)
    make_pivot_one(A,B,i)
    print(A)
    subtract_rows(A,B,i)
    print(A)

print(A)
print(B)