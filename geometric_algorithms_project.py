from copy import deepcopy
from operator import indexOf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Vertex:
    x: float
    y: float
    index: int

@dataclass
class Edge:
    start_vertex: Vertex
    end_vertex: Vertex

@dataclass
class Vertex_With_Neighbor:
    vertex: Vertex
    prev_neighbor_index: int
    next_neighbor_index: int


vertices = np.loadtxt("ExampleInput.txt", dtype=np.float32)
#editable_vertices = deepcopy(vertices)
editable_indices = np.arange(len(vertices))
permuted_indices = np.random.permutation(editable_indices)

## Store the points with neighbors that we will cut
vertices_with_neighbors = []

## Stop at the last 3 points
for index in permuted_indices[:-3]:
    
    modified_index = np.where(editable_indices == index)
    # if modified_index:
    #     print(f"index: {index}")
    #     print(modified_index[0][0])
    #     print(editable_indices)
    #     print()
    
    ## Create the vertex
    vertex = Vertex(vertices[index][0], vertices[index][1], index=index)

    ## Get the indices of the previous and the next neighbours with wrapping
    prev_neighbor_index = editable_indices[-1] if modified_index[0][0] - 1 == -1 else editable_indices[modified_index[0][0] - 1]
    next_neighbor_index = editable_indices[0] if modified_index[0][0] + 1 == len(editable_indices) else editable_indices[modified_index[0][0] + 1]

    ## Create the vertex with neighbor
    vertex_with_neighbor = Vertex_With_Neighbor(vertex, prev_neighbor_index, next_neighbor_index)
    
    ## Append it to the result list
    vertices_with_neighbors.append(vertex_with_neighbor)

    ## For debugging purposes
    # print(f"index: {index}")
    # print(f"modified_index: {modified_index[0][0]}")
    # print(f"editable_indices: {editable_indices}")
    # print(f"prev_neighbor_index: {prev_neighbor_index}")
    # print(f"next_neighbor_index: {next_neighbor_index}")
    # print(f"vertices_with_neighbors: {vertices_with_neighbors}")
    # print()

    ## Remove the index from the editable indices list
    editable_indices = np.delete(editable_indices, modified_index)

for index in permuted_indices[-3:]:
    
    modified_index = np.where(editable_indices == index)
    # if modified_index:
    #     print(f"index: {index}")
    #     print(modified_index[0][0])
    #     print(editable_indices)
    #     print()
    
    ## Create the vertex
    vertex = Vertex(vertices[index][0], vertices[index][1], index=index)

    ## Get the indices of the previous and the next neighbours with wrapping
    prev_neighbor_index = editable_indices[-1] if modified_index[0][0] - 1 == -1 else editable_indices[modified_index[0][0] - 1]
    next_neighbor_index = editable_indices[0] if modified_index[0][0] + 1 == len(editable_indices) else editable_indices[modified_index[0][0] + 1]

    ## Create the vertex with neighbor
    vertex_with_neighbor = Vertex_With_Neighbor(vertex, prev_neighbor_index, next_neighbor_index)
    
    ## Append it to the result list
    vertices_with_neighbors.append(vertex_with_neighbor)

final_edge_list = []

#counter = 0
## Get Initial Triangle
prev_vertex = vertices_with_neighbors[-1].vertex
for vertex_with_neighbor in reversed(vertices_with_neighbors[-3:-1]):
    cur_vertex = vertex_with_neighbor.vertex
    print(f"vertex_with_neighbor: {vertex_with_neighbor}")
    edge = Edge(prev_vertex, cur_vertex)
    print(f"edge: {edge}")
    final_edge_list = final_edge_list + [edge]
    print(f"final_edge_list: {final_edge_list}")
    prev_vertex = cur_vertex
    fig, ax = plt.subplots()
    edges_for_plotting = []
    edges_for_plotting = [[(i.start_vertex.x, i.start_vertex.y), (i.end_vertex.x, i.end_vertex.y)] for i in final_edge_list]
    #lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    lc = mc.LineCollection(edges_for_plotting, colors=(0.3, 0.3, 0.6, 1), linewidths=2)
    #lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
    ax.add_collection(lc)
    #ax.add_collection(lc2)
    ax.autoscale()
    plt.show()
    #counter += 1
    #if counter == 2:
        # edge = Edge(cur_vertex, vertices_with_neighbors[-1].vertex)
        # final_edge_list = final_edge_list + [edge]

edge = Edge(cur_vertex, vertices_with_neighbors[-1].vertex)
final_edge_list = final_edge_list + [edge]

## Plot Initial Triangle
fig, ax = plt.subplots()
edges_for_plotting = []
edges_for_plotting = [[(i.start_vertex.x, i.start_vertex.y), (i.end_vertex.x, i.end_vertex.y)] for i in final_edge_list]
#lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
lc = mc.LineCollection(edges_for_plotting, colors=(0.3, 0.3, 0.6, 1), linewidths=2)
#lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
ax.add_collection(lc)
#ax.add_collection(lc2)
ax.autoscale()
plt.show()



# x_coords_deleted = [i.vertex.x for i in vertices_with_neighbors]
# y_coords_deleted = [i.vertex.y for i in vertices_with_neighbors]

# x_coords_starting = [vertices[i][0] for i in editable_indices]
# y_coords_starting = [vertices[i][1] for i in editable_indices]

# plotting_coordinates_x = x_coords_starting + [0, 1, 2]
# plotting_coordinates_y = y_coords_starting + [1, 2, 3]

# plt.plot(plotting_coordinates_x, plotting_coordinates_y)
# plt.show()

# plt.plot(vertices_with_neighbors[:], vertices_with_neighbors[:, 1])
# plt.plot(np.append(vertices[:, 0], vertices[0, 0]), np.append(vertices[:, 1], vertices[0, 1]))
# plt.plot(np.append(x_coords_starting, x_coords_starting[0]), np.append(y_coords_starting, y_coords_starting[0]))

# zipped_coords = [(i, j) for i, j in zip(x_coords_starting, y_coords_starting)]
# starting_edges = []
# for i in range(len(zipped_coords)-1):
#     starting_edges = starting_edges + [[zipped_coords[i], zipped_coords[i+1]]]
# starting_edges = starting_edges + [[starting_edges[-1][1], starting_edges[0][0]]]
#print(lines)

# starting_vertices = []
# for starting_index in editable_indices:
#     vertex = Vertex(vertices[starting_index][0], vertices[starting_index][1], index = starting_index)
#     starting_vertices = starting_vertices.append(vertex)
# final_edge_list = []


## To plot the edges
# fig, ax = plt.subplots()
# #lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
# lc = mc.LineCollection(starting_edges, colors=(0.0, 0.75, 0.0, 1), linewidths=2)
# #lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
# ax.add_collection(lc)
# #ax.add_collection(lc2)
# ax.autoscale()
# plt.show()
# for vertex in vertices_permuted
#

# for index in permuted_indices:

# print(vertices_with_neighbors)
#print(len(vertices))
#print(vertices_permuted)
# print(vertices)
# print(vertices_permutated)
# print(vertices_permuted[:, 0])
# print(vertices_permuted[:, 1])
