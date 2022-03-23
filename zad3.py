import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visit(T, matrix, visited, v, s, t):
    visited[v] = True

    if v != s and v != t:
        for i in range(len(T)):
            if T[v][i] > 0:
                matrix[v][v] += 1.0 / T[v][i]
                if i != v:
                    matrix[v][i] -= 1.0 / T[v][i]

    for i in range(len(T)):
        if T[v][i] > 0 and not visited[i]:
            visit(T, matrix, visited, i, s, t)

def test(current):
    for i in range(len(current)):
        sum=0
        for j in range(len(current[i])):
            sum+=current[i][j]
        if round(sum,2)!=0.00:
            print(round(sum,2))
            return False
    return True


resistance_graph = [[-1, 1, 3, 2, -1],
                    [1, -1, 2, -1, 1],
                    [3, 2, -1, 3, -1],
                    [2, -1, 3, -1, 2],
                    [-1, 1, -1, 2, -1]]

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

visit(resistance_graph, matrix, visited, 0, s, t)

solution = np.linalg.solve(matrix, right_vector)

print(solution)

current = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

for i in range(len(solution)):
    for j in range(len(solution)):
        if resistance_graph[i][j]!=-1:
            current[i][j] = (solution[j] - solution[i]) / resistance_graph[i][j]

for row in current:
    print(row)

print(test(current))

circuit = nx.DiGraph()

for i in range(len(current)):
    circuit.add_node(i)

for i in range(len(current)):
    for j in range(len(current)):
        if resistance_graph[i][j]!=-1:
            if current[i][j] < 0:
                circuit.add_edge(i, j, weight=round((-1) * current[i][j], 2))
            elif current[i][j] > 0:
                circuit.add_edge(j, i, weight=round(current[i][j], 2))

pos = nx.spring_layout(circuit)
nx.draw_networkx(circuit, pos)
edge_labels = dict([((u, v,), d['weight']) for u, v, d in circuit.edges(data=True)])
nx.draw_networkx_edge_labels(circuit, pos, edge_labels=edge_labels)
plt.show()
