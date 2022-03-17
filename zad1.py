import numpy as np
import time
import random

def scale(A, B):
    for i in range(len(A)):
        max_el = 0
        for el in A[i]:
            if abs(el) > max_el:
                max_el = abs(el)
        A[i] = [el / max_el for el in A[i]]
        B[i] /= max_el


def partial_pivot(A, col):
    pivot_val = abs(A[col][col])
    pivot_index = col
    if col < len(A) - 1:
        for j in range(col + 1, len(A)):
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
        A[i][j] = A[i][j] / pivot
    B[i] = B[i] / pivot


def subtract_rows(A, B, i):
    # for pivot==1
    for j in range(len(A)):
        if j != i:
            temp = [el * A[j][i] for el in A[i]]
            B[j] -= B[i] * A[j][i]
            A[j] = np.subtract(A[j], temp)


def solve(A, B):
    scale(A, B)
    for i in range(len(A[0])):
        pivot_index = partial_pivot(A, i)
        if i != pivot_index:
            swap_rows(A, B, i, pivot_index)
        make_pivot_one(A, B, i)
        subtract_rows(A, B, i)

    return B

def generate_matrix(size):
    A = np.array([[random.uniform(-500.0,500.0) for _ in range(size)]for _ in range(size)])
    B = [random.uniform(-500.0,500.0) for _ in range(size)]
    return A, B



A,B=generate_matrix(1000)
start_t=time.time()
np.linalg.solve(A, B)
end_t=time.time()
print("numpy solver: ", end_t-start_t)

start_t=time.time()
solve(A, B)
end_t=time.time()
print("my solver: ", end_t-start_t)