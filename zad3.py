import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def parse_file(name):
    f=open(name)
    input=f.read().split()
    s=int(input[0])
    t=int(input[1])
    E=float(input[2])

    list2=[]
    for i in range(3,len(input)):
        list2.append(input[i][1:-1].replace('(','').replace(')','').split(','))
        for j in range(0,len(list2[i-3]),2):
            if list2[i-3][j]=='':
                list2[i - 3].remove('')
            else:
                list2[i-3][j]=int(list2[i-3][j])
                list2[i - 3][j+1] = float(list2[i - 3][j+1])


    resistance_graph=[[-1 for _ in range(len(list2))]for _ in range(len(list2))]
    for i in range(len(list2)):
        for j in range(0,len(list2[i]),2):
            if resistance_graph[i][list2[i][j]]==resistance_graph[list2[i][j]][i]==-1:
                resistance_graph[i][list2[i][j]]=resistance_graph[list2[i][j]][i]=list2[i][j+1]

    return s,t,E,resistance_graph

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

def solve_circuit(filename):
    s,t,E,resistance_graph=parse_file(filename)

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
    else:
        print("Test passed")

    circuit = nx.DiGraph()

    for i in range(len(current)):
        circuit.add_node(i)

    for i in range(len(current)):
        for j in range(len(current)):
            if resistance_graph[i][j]!=-1:
                if not circuit.has_edge(i,j) and not circuit.has_edge(j,i):
                    if current[i][j] < 0:
                        circuit.add_edge(i, j, weight=round((-1) * current[i][j], 2))
                    elif current[i][j] > 0:
                        circuit.add_edge(j, i, weight=round(current[i][j], 2))

    pos = nx.spring_layout(circuit)
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in circuit.edges(data=True)])
    edges,weights = zip(*nx.get_edge_attributes(circuit,'weight').items())
    nx.draw(circuit, pos, edge_color=weights,edge_cmap=plt.cm.YlOrRd, width=1, node_size=5)
    #nx.draw_networkx_edge_labels(circuit, pos, edge_labels=edge_labels)
    #nx.draw_networkx_labels(circuit, pos)
    plt.show()

def generate_graphs():
    graphs=[]
    graphs.append(nx.erdos_renyi_graph(20, 0.5))
    graphs.append(nx.erdos_renyi_graph(100, 0.1))
    graphs.append(nx.cubical_graph())
    graphs.append(nx.barbell_graph(10,10))
    graphs.append(nx.newman_watts_strogatz_graph(50, 10, 0.05))
    #graphs.append(nx.grid_graph(dim=(10,10)))

    for i in range(len(graphs)):
        f = open("graph"+str(i)+".txt", "w")
        f.write("0 1 10000 ")
        for j in range(len(graphs[i].nodes)):
            f.write("(")
            if graphs[i].edges(j):
                for k in range(len(graphs[i].edges(j))):
                    f.write("("+str(list(graphs[i].edges(j))[k][1])+","+str(round(random.uniform(100, 2000),2))+")")
                    if k!=len(graphs[i].edges(j))-1:
                        f.write(",")
            f.write(") ")

generate_graphs()
solve_circuit("graph0.txt")
solve_circuit("graph1.txt")
solve_circuit("graph2.txt")
solve_circuit("graph3.txt")