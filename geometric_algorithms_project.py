import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Point_With_Neighbor:
    point: Point
    prev_neighbor_index: int
    next_neighbor_index: int

# testpoint = Point(1,2)

# p = Point_With_Neighbor(testpoint, 2, 3)
# print(testpoint)
# print(testpoint.x)
# print(p)  # Point(x=1.5, y=2.5, z=0.0)

vertices = np.loadtxt("ExampleInput.txt", dtype=np.float32)
permuted_indices = np.random.permutation(np.arange(len(vertices)))
#print(len(vertices))
#print(vertices_permuted)
# print(vertices)
# print(vertices_permutated)
# print(vertices_permuted[:, 0])
# print(vertices_permuted[:, 1])
# plt.plot(vertices_permuted[:, 0], vertices_permuted[:, 1])
# plt.plot(np.append(vertices[:, 0], vertices[0, 0]), np.append(vertices[:, 1], vertices[0, 1]))
# plt.show()


# for vertex in vertices_permuted


for index in permuted_indices:
    
