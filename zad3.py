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

def test(current,s,t):
    for i in range(len(current)):
        if i!=s and i!=t:
            sum=0
            for j in range(len(current[i])):
                sum+=current[i][j]
            if round(sum,2)!=0.00:
                print(round(sum,2))
                return False
    return True

def parse_file(name):
    f=open(name)
    input=f.read().split()
    print(input)
    s=int(input[0])
    t=int(input[1])
    E=float(input[2])

    list2=[]
    for i in range(3,len(input)):
        list2.append(input[i][1:-1].replace('(','').replace(')','').split(','))
        for j in range(len(list2[i-3])):
            if list2[i-3][j]=='':
                list2[i - 3].remove('')
            else:
                list2[i-3][j]=int(list2[i-3][j])


    resistance_graph=[[-1 for _ in range(len(list2))]for _ in range(len(list2))]
    for i in range(len(list2)):
        for j in range(0,len(list2[i]),2):
            if  resistance_graph[i][list2[i][j]]==resistance_graph[list2[i][j]][i]==-1:
                resistance_graph[i][list2[i][j]]=resistance_graph[list2[i][j]][i]=list2[i][j+1]

    return s,t,E,resistance_graph

s,t,E,resistance_graph=parse_file("resistance.txt")

right_vector = [0 for _ in range(len(resistance_graph))]
matrix = [[0 for _ in range(len(resistance_graph[0]))] for _ in range(len(resistance_graph))]
visited = [False for _ in range(len(resistance_graph))]

matrix[s][s] = 1
right_vector[s] = E
matrix[t][t] = 1
right_vector[t] = 0

visit(resistance_graph, matrix, visited, 0, s, t)

solution = np.linalg.solve(matrix, right_vector)

current = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

for i in range(len(solution)):
    for j in range(len(solution)):
        if resistance_graph[i][j]!=-1:
            current[i][j] = (solution[j] - solution[i]) / resistance_graph[i][j]


if not test(current,s,t):
    raise Exception("Error")

circuit = nx.DiGraph()

for i in range(len(current)):
    circuit.add_node(i)

for i in range(len(current)):
    for j in range(len(current)):
        if resistance_graph[i][j]!=-1:
            if not circuit.has_edge(i,j) and not circuit.has_edge(j,i):
                if current[i][j] < 0:
                    circuit.add_edge(i, j,color='r', weight=round((-1) * current[i][j], 2))
                elif current[i][j] > 0:
                    circuit.add_edge(j, i,color='r', weight=round(current[i][j], 2))

pos = nx.spring_layout(circuit)
edge_labels = dict([((u, v,), d['weight']) for u, v, d in circuit.edges(data=True)])
edges,weights = zip(*nx.get_edge_attributes(circuit,'weight').items())
nx.draw(circuit, pos, edge_color=weights,edge_cmap=plt.cm.YlOrRd, width=2.5)
nx.draw_networkx_edge_labels(circuit, pos, edge_labels=edge_labels)
nx.draw_networkx_labels(circuit, pos)
plt.show()
