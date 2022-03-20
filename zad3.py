import numpy as np

def visit(T, matrix, visited, v, s, t):
    visited[v] = True

    if v != s and v != t:
        for i in range(len(T)):
            if T[v][i] > 0:
                matrix[v][v] += 1/T[v][i]
                if i != v:
                    matrix[v][i] -= 1/T[v][i]

    for i in range(len(T)):
        if T[v][i] > 0 and not visited[i]:
            visit(T, matrix, visited, i, s, t)


resistance_graph = [[-1, 1, 3, 2],
                    [1, -1, 2,-1],
                    [3, 2, -1,3],
                    [2,-1,3,-1]]

s = 0
t = 2
E = 5

right_vector = [0 for _ in range(len(resistance_graph))]
matrix = [[0 for _ in range(len(resistance_graph[0]))] for _ in range(len(resistance_graph))]
visited = [False for _ in range(len(resistance_graph))]

matrix[s][s] = 1
right_vector[s] = E
matrix[t][t] = 1
right_vector[t] = 0

visit(resistance_graph, matrix, visited, s, s, t)

solution = np.linalg.solve(matrix,right_vector)

print(solution)

current = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

for i in range(len(solution)):
    for j in range(i+1, len(solution)):
        if solution[i] > 0 and solution[i] > 0:
            current[i][j] = (solution[j]-solution[i])/resistance_graph[i][j]

for row in current:
    print(row)
