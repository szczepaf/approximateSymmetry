import annealing_versions.sa_eigenvector as sa_eigenvector
import networkx as nx


# BA graph
n = 50
k = 3
A = nx.barabasi_albert_graph(n, k)
A = nx.to_numpy_array(A)

permutation, energy = sa_eigenvector.annealing(A, steps=30000)
print(permutation)
print(energy)
