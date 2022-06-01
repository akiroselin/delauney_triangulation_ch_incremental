import numpy as np

vertexes = np.loadtxt("ExampleInput.txt", dtype=np.float32)
print(vertexes)
vertexes_permutated = np.random.permutation(vertexes)
print(vertexes_permutated)
vertexes_permutated_index = [np.where(vertexes==i)[0][0] for i in vertexes_permutated]
print(vertexes_permutated_index)
vertexes_permutated_neighbours = [[vertexes[i-1],vertexes[(i+1)%7]] for i in vertexes_permutated_index]
print(vertexes_permutated_neighbours)
