import numpy as np

vertexes = np.loadtxt("ExampleInput.txt", dtype=np.float32)
print(vertexes)
vertexes_permutated = np.random.permutation(vertexes)
print(vertexes_permutated)