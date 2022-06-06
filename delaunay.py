from copy import deepcopy
from operator import indexOf
from random import random, seed
from wsgiref.validate import WriteWrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from dataclasses import dataclass
from copy import deepcopy
from typing import Any

#np.random.seed(3)
@dataclass
class Vertex:
    x: float
    y: float
    index: int
    inc_edge: Any #Type Any because the Half Edge isn't defined yet at this point

@dataclass
class Half_Edge:
    origin_vertex: Vertex
    twin: Any
    inc_face: Any #left face
    next_edge: Any #Type Any because the Half Edge isn't defined yet at this point
    prev_edge: Any #Type Any because the Half Edge isn't defined yet at this point

@dataclass
class Face:
    inc_edge: Half_Edge

@dataclass
class Vertex_Index_With_Neighbor:
    vertex_index: int
    prev_neighbor_index: int
    next_neighbor_index: int

final_edge_list = []
final_face_list = []
## Read the input file. Vertices are already in convex position
vertices = np.loadtxt("ExampleInput.txt", dtype=np.float32)
#print(f"vertices: {vertices}")
# vertices = np.array([[1, 1],
#  [1, 3],
#  [3, 3],
#  [3, 2.5],
# ])
# print(vertices)
## Create a copy of the vertices array to edit them to be able to store the 
editable_indices = np.arange(len(vertices))

## Randomly permute the indices 
permuted_indices = np.random.permutation(editable_indices)
#print(f"permuted: {permuted_indices}")
#permuted_indices = deepcopy(editable_indices)

## Store the points with neighbors that we will cut
vertex_indices_with_neighbors = []

## Stop at the last 3 points
for index in permuted_indices[:-3]:
    
    modified_index = np.where(editable_indices == index)

    ## Get the indices of the previous and the next neighbors with wrapping
    prev_neighbor_index = editable_indices[-1] if modified_index[0][0] - 1 == -1 else editable_indices[modified_index[0][0] - 1]
    next_neighbor_index = editable_indices[0] if modified_index[0][0] + 1 == len(editable_indices) else editable_indices[modified_index[0][0] + 1]

    ## Create the vertex with neighbor
    vertex_index_with_neighbor = Vertex_Index_With_Neighbor(index, prev_neighbor_index, next_neighbor_index)
    
    ## Append it to the result list
    vertex_indices_with_neighbors.append(vertex_index_with_neighbor)

    # ## For debugging purposes
    # print(f"index: {index}")
    # print(f"modified_index: {modified_index[0][0]}")
    # print(f"editable_indices: {editable_indices}")
    # print(f"prev_neighbor_index: {prev_neighbor_index}")
    # print(f"next_neighbor_index: {next_neighbor_index}")
    # print(f"vertices_with_neighbors: {vertex_indices_with_neighbors}")
    # print()

    ## Remove the index from the editable indices list
    editable_indices = np.delete(editable_indices, modified_index)

for index in permuted_indices[-3:]:
    
    modified_index = np.where(editable_indices == index)

    ## Get the indices of the previous and the next neighbors with wrapping
    prev_neighbor_index = editable_indices[-1] if modified_index[0][0] - 1 == -1 else editable_indices[modified_index[0][0] - 1]
    next_neighbor_index = editable_indices[0] if modified_index[0][0] + 1 == len(editable_indices) else editable_indices[modified_index[0][0] + 1]

    ## Create the vertex with neighbor
    vertex_with_neighbor = Vertex_Index_With_Neighbor(index, prev_neighbor_index, next_neighbor_index)
    
    ## Append it to the result list
    vertex_indices_with_neighbors.append(vertex_with_neighbor)

    # ## For debugging purposes
    # print(f"index: {index}")
    # print(f"modified_index: {modified_index[0][0]}")
    # print(f"editable_indices: {editable_indices}")
    # print(f"prev_neighbor_index: {prev_neighbor_index}")
    # print(f"next_neighbor_index: {next_neighbor_index}")
    # print(f"vertices_with_neighbors: {vertex_indices_with_neighbors}")
    # print()


#####################################################################################
# We now have the vertex indices with their respective neighbors at the time of 
# deletion. We will proceed by building the DCEL representing the Delaunay 
# Triangulation one triangle at a time, making sure to do the necessary edge flips.
#####################################################################################

# initial_edge_list = []

dcel_vertex_list = [None] * len(vertices)

## Get the first vertex that we're considering. From it, we will have all the necessary information to build the initial triangle
vertex_index_with_neighbor = vertex_indices_with_neighbors[-1]
#print(f"last index: {vertex_indices_with_neighbors[-1]}")
#print(f"second to last: {vertex_indices_with_neighbors[-2]}")
#print(f"3rd to last: {vertex_indices_with_neighbors[-3]}")

## Get the indices
cur_vertex_index = vertex_index_with_neighbor.vertex_index
next_vertex_index = vertex_index_with_neighbor.next_neighbor_index
prev_vertex_index = vertex_index_with_neighbor.prev_neighbor_index

## Create the vertices with the info currently available.
dcel_vertex_list[cur_vertex_index] = Vertex(x=vertices[cur_vertex_index][0], y=vertices[cur_vertex_index][1], index=cur_vertex_index,inc_edge=None)

dcel_vertex_list[next_vertex_index] = Vertex(x=vertices[next_vertex_index][0], y=vertices[next_vertex_index][1], index=next_vertex_index,inc_edge=None)

dcel_vertex_list[prev_vertex_index] = Vertex(x=vertices[prev_vertex_index][0], y=vertices[prev_vertex_index][1], index=prev_vertex_index,inc_edge=None)

cur_inc_edge = Half_Edge(inc_face=None, next_edge=None, prev_edge=None, 
origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, 
next_edge=None,origin_vertex=dcel_vertex_list[next_vertex_index], prev_edge=None, twin=None))
cur_inc_edge.twin.twin = cur_inc_edge

next_inc_edge = Half_Edge(inc_face=None, next_edge=None, prev_edge=cur_inc_edge, 
origin_vertex=dcel_vertex_list[next_vertex_index], twin=Half_Edge(inc_face=None, 
next_edge=None,origin_vertex=dcel_vertex_list[prev_vertex_index], prev_edge=None, twin=None))
next_inc_edge.twin.twin=next_inc_edge

prev_inc_edge = Half_Edge(inc_face=None, next_edge=cur_inc_edge, prev_edge=next_inc_edge, 
origin_vertex=dcel_vertex_list[prev_vertex_index], twin=Half_Edge(inc_face=None, 
next_edge=None,origin_vertex=dcel_vertex_list[cur_vertex_index], prev_edge=None, twin=None))
prev_inc_edge.twin.twin=prev_inc_edge

## Fill up the remaining information for the half edges
cur_inc_edge.next_edge = next_inc_edge
cur_inc_edge.prev_edge = prev_inc_edge
next_inc_edge.next_edge = prev_inc_edge

## Fill up the remaining information for the vertices
dcel_vertex_list[cur_vertex_index].inc_edge = cur_inc_edge
dcel_vertex_list[next_vertex_index].inc_edge = next_inc_edge
dcel_vertex_list[prev_vertex_index].inc_edge = prev_inc_edge

## Create a face
face = Face(inc_edge=cur_inc_edge)

## Add it to the edges
dcel_vertex_list[cur_vertex_index].inc_edge.inc_face = face
dcel_vertex_list[next_vertex_index].inc_edge.inc_face = face
dcel_vertex_list[prev_vertex_index].inc_edge.inc_face = face

#print(f"Initial indices: ")
#print(vertex_indices_with_neighbors[-1])
#print(vertex_indices_with_neighbors[-2])
#print(vertex_indices_with_neighbors[-3])

# edge = dcel_vertex_list[vertex_indices_with_neighbors[-1]].inc_edge
# counter = 0
# while counter < 10:
#     print(f"next: edge1 from {edge.origin_vertex.}")

## We now have the initial triangle. We start with adding the other vertices 
# one by one and doing the inCircle Test, before deciding if we flip or not.

# edge_list_for_plotting = [[(1,2), (2,3)], [(1,2), (2,3)]]
# edge_list_for_plotting = [[(vertices[cur_vertex_index][0], vertices[cur_vertex_index][1]), (vertices[next_vertex_index][0], vertices[next_vertex_index][1])], 
# [(vertices[next_vertex_index][0], vertices[next_vertex_index][1]), (vertices[prev_vertex_index][0], vertices[prev_vertex_index][1])],
# [(vertices[prev_vertex_index][0], vertices[prev_vertex_index][1]), (vertices[cur_vertex_index][0], vertices[cur_vertex_index][1])]]
# fig, ax = plt.subplots()
#edge_list_for_plotting = []
#lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
# lc = mc.LineCollection(edge_list_for_plotting, colors=(0.0, 0.75, 0.0, 1),linewidths=2)
# ax.add_collection(lc)
# ax.autoscale()
# plt.show()
## d is the new point
## a is d.next (neighbor)
## b is a.next (next vertex around the incident face)
## c is d.prev == b.next == a.next.next
def in_circle_test(a_x: float, a_y: float,
                 b_x: float, b_y: float,
                 c_x: float, c_y: float,
                 d_x: float, d_y: float,):
    matrix = np.array([[a_x, a_y, a_x**2 + a_y**2, 1],
                       [b_x, b_y, b_x**2 + b_y**2, 1],
                       [c_x, c_y, c_x**2 + c_y**2, 1],
                       [d_x, d_y, d_x**2 + d_y**2, 1]])
    
    det = np.linalg.det(matrix)

    if(det > 0):
        return True
    else:
        return False

def perform_initial_flip(cur_vertex_index, prev_vertex_index, next_vertex_index, third_vertex_index):

    new_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge, prev_edge=dcel_vertex_list[prev_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=dcel_vertex_list[prev_vertex_index].inc_edge.next_edge, origin_vertex=dcel_vertex_list[third_vertex_index], twin=None))
    new_edge.twin.twin = new_edge

    new_next_edge = Half_Edge(inc_face=None, next_edge=new_edge.twin.prev_edge, prev_edge=new_edge.twin, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None,next_edge=None,prev_edge=None,origin_vertex=dcel_vertex_list[next_vertex_index],twin=None))
    new_next_edge.twin.twin = new_next_edge

    dcel_vertex_list[prev_vertex_index].inc_edge.twin.origin_vertex = dcel_vertex_list[cur_vertex_index]
    dcel_vertex_list[prev_vertex_index].inc_edge.next_edge = new_edge
    new_edge.twin.next_edge = new_next_edge
    dcel_vertex_list[cur_vertex_index].inc_edge = new_next_edge

    new_edge.next_edge.prev_edge = new_edge
    new_edge.prev_edge.next_edge = new_edge
    new_edge.twin.next_edge.prev_edge = new_edge.twin
    new_edge.twin.prev_edge.next_edge = new_edge.twin

    face1 = Face(inc_edge=new_edge)
    face2 = Face(inc_edge=new_edge.twin)

    new_edge.inc_face = face1
    new_edge.prev_edge.inc_face = face1
    new_edge.next_edge.inc_face = face1

    new_edge.twin.inc_face = face2
    new_edge.twin.prev_edge.inc_face = face2
    new_edge.twin.next_edge.inc_face = face2

    return

def perform_inside_flip_left(cur_vertex_index, prev_vertex_index, next_vertex_index, third_vertex_index):

    new_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge, prev_edge=dcel_vertex_list[prev_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=dcel_vertex_list[next_vertex_index].inc_edge.prev_edge, prev_edge=dcel_vertex_list[next_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[third_vertex_index], twin=None))
    new_edge.twin.twin = new_edge

    new_edge.next_edge.prev_edge = new_edge
    new_edge.next_edge.next_edge = new_edge.prev_edge
    new_edge.prev_edge.next_edge = new_edge
    new_edge.prev_edge.prev_edge = new_edge.next_edge

    new_edge.twin.next_edge.prev_edge = new_edge.twin
    new_edge.twin.next_edge.next_edge = new_edge.twin.prev_edge
    new_edge.twin.prev_edge.next_edge = new_edge.twin
    new_edge.twin.prev_edge.prev_edge = new_edge.twin.next_edge
    new_edge.next_edge.twin.prev_edge.origin_vertex = dcel_vertex_list[third_vertex_index]

    face1 = Face(inc_edge=new_edge)
    face2 = Face(inc_edge=new_edge.twin)

    new_edge.inc_face = face1
    new_edge.prev_edge.inc_face = face1
    new_edge.next_edge.inc_face = face1

    new_edge.twin.inc_face = face2
    new_edge.twin.prev_edge.inc_face = face2
    new_edge.twin.next_edge.inc_face = face2
    return

def perform_inside_flip_right(cur_vertex_index, prev_vertex_index, next_vertex_index, third_vertex_index):

    new_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.prev_edge, prev_edge=dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.next_edge, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=dcel_vertex_list[cur_vertex_index].inc_edge, prev_edge=dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.next_edge, origin_vertex=dcel_vertex_list[third_vertex_index], twin=None))
    new_edge.twin.twin = new_edge
    #dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge = new_edge.twin

    #dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge = new_edge
    #new_edge.twin.next_edge = new_next_edge
    # dcel_vertex_list[cur_vertex_index].inc_edge = new_next_edge

    new_edge.next_edge.prev_edge = new_edge
    new_edge.next_edge.next_edge = new_edge.prev_edge
    new_edge.next_edge.next_edge.twin.origin_vertex = dcel_vertex_list[cur_vertex_index]
    new_edge.prev_edge.next_edge = new_edge
    new_edge.prev_edge.prev_edge = new_edge.next_edge

    new_edge.twin.next_edge.prev_edge = new_edge.twin
    new_edge.twin.next_edge.next_edge = new_edge.twin.prev_edge
    new_edge.twin.prev_edge.next_edge = new_edge.twin
    new_edge.twin.prev_edge.prev_edge = new_edge.twin.next_edge

    face1 = Face(inc_edge=new_edge)
    face2 = Face(inc_edge=new_edge.twin)

    new_edge.inc_face = face1
    new_edge.prev_edge.inc_face = face1
    new_edge.next_edge.inc_face = face1

    new_edge.twin.inc_face = face2
    new_edge.twin.prev_edge.inc_face = face2
    new_edge.twin.next_edge.inc_face = face2

## We start inserting points in reverse order of deletion
for vertex_index_with_neighbor in reversed(vertex_indices_with_neighbors[:-3]):
    
    print(f"Vertex index: {vertex_index_with_neighbor.vertex_index}")
    ## Get the indices
    cur_vertex_index = vertex_index_with_neighbor.vertex_index
    next_vertex_index = vertex_index_with_neighbor.next_neighbor_index
    prev_vertex_index = vertex_index_with_neighbor.prev_neighbor_index
    third_vertex_index = dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.origin_vertex.index

    next_vertex_index_archive = next_vertex_index
    prev_vertex_index_archive = prev_vertex_index
    third_vertex_index_archive = third_vertex_index

    ## Create the new vertex
    dcel_vertex_list[cur_vertex_index] = Vertex(inc_edge=None, index=cur_vertex_index, x=vertices[cur_vertex_index][0], y=vertices[cur_vertex_index][1])

    ## Check if we need to perform a flip
    need_to_flip = in_circle_test(d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                 a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                 b_x=vertices[third_vertex_index][0], b_y=vertices[third_vertex_index][1],
                 c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1])
    initial_flip = True

    ## In case the point is inside the circle, we do the necessary updates, and propagate.

    if need_to_flip:

        if initial_flip:
            print("Initial Flip")
            perform_initial_flip(cur_vertex_index=cur_vertex_index, prev_vertex_index=prev_vertex_index, next_vertex_index=next_vertex_index, third_vertex_index=third_vertex_index)
            initial_flip = False
            third_right = third_vertex_index
            third_left = third_vertex_index

        while dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.prev_edge and need_to_flip:
            print("Trying to flip right")
            prev_right = third_right
            third_right = dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.prev_edge.origin_vertex.index
            need_to_flip = in_circle_test(d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                b_x=vertices[third_right][0], b_y=vertices[third_right][1],
                c_x=vertices[prev_right][0], c_y=vertices[prev_right][1])
            if need_to_flip:
                print("Flipping right")
                perform_inside_flip_right(cur_vertex_index=cur_vertex_index, prev_vertex_index=prev_right, next_vertex_index=next_vertex_index, third_vertex_index=third_right)
        need_to_flip = True
        while dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.twin.prev_edge and need_to_flip:
            print("Trying to flip left")
            next_left = third_left
            third_left = dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.twin.prev_edge.origin_vertex.index
            need_to_flip = in_circle_test(d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                a_x=vertices[next_left][0], a_y=vertices[next_left][1],
                b_x=vertices[third_left][0], b_y=vertices[third_left][1],
                c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1])
            if need_to_flip:
                print("Flipping Left")
                perform_inside_flip_left(cur_vertex_index=cur_vertex_index, prev_vertex_index=prev_vertex_index, next_vertex_index=next_left, third_vertex_index=third_left)


    else:

        dcel_vertex_list[cur_vertex_index].inc_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[prev_vertex_index].inc_edge.twin, prev_edge=None, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=None, origin_vertex=dcel_vertex_list[next_vertex_index], twin=None))
        dcel_vertex_list[cur_vertex_index].inc_edge.twin.twin = dcel_vertex_list[cur_vertex_index].inc_edge

        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge = Half_Edge(inc_face=None, prev_edge=dcel_vertex_list[cur_vertex_index].inc_edge.next_edge, next_edge=dcel_vertex_list[cur_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[prev_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=None, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=None))
        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge.twin.twin = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge

        dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.next_edge = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge

        dcel_vertex_list[prev_vertex_index].inc_edge = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge

        face = Face(inc_edge=dcel_vertex_list[cur_vertex_index].inc_edge)

        dcel_vertex_list[cur_vertex_index].inc_edge.inc_face = face
        dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.inc_face = face
        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge.inc_face = face

    edge_list_for_plotting = []
    for vertex in dcel_vertex_list:
        if vertex != None:
            edge = vertex.inc_edge
            
            # edge_for_plot1 = [(vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1]), (vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1])]
            # edge_for_plot2 = [(vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1]), (vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1])]
            # edge_for_plot3 = [(vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1]), (vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1])]

            edge_for_plot1 = [(vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1]), (vertices[edge.twin.origin_vertex.index][0], vertices[edge.twin.origin_vertex.index][1])]
            edge_for_plot2 = [(vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1]), (vertices[edge.next_edge.twin.origin_vertex.index][0], vertices[edge.next_edge.twin.origin_vertex.index][1])]
            edge_for_plot3 = [(vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1]), (vertices[edge.prev_edge.twin.origin_vertex.index][0], vertices[edge.prev_edge.twin.origin_vertex.index][1])]

            # edge_for_plot1 = [(vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1]), (vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1])]
            # edge_for_plot2 = [(vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1]), (vertices[edge.next_edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.next_edge.origin_vertex.index][1])]
            # edge_for_plot3 = [(vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1]), (vertices[edge.prev_edge.next_edge.origin_vertex.index][0], vertices[edge.prev_edge.next_edge.origin_vertex.index][1])]

            edge_list_for_plotting = edge_list_for_plotting + [edge_for_plot1, edge_for_plot2, edge_for_plot3]
            # edge = edge.next_edge

    color_for_plotting = (0.3, 0.3, 0.6, 1)
    x_coords = [i[0] for i in vertices]
    y_coords = [i[1] for i in vertices]
    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, c=color_for_plotting)
    for vertex_index in vertex_indices_with_neighbors:
        ax.annotate(vertex_index.vertex_index, (vertices[vertex_index.vertex_index][0], vertices[vertex_index.vertex_index][1]), xytext=(vertices[vertex_index.vertex_index][0] + 0.015, vertices[vertex_index.vertex_index][1]), c=color_for_plotting)

    lc = mc.LineCollection(edge_list_for_plotting, colors=(0.0, 0.75, 0.0, 1),linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    plt.show()


def define_center(p1, p2, p3):
    """
    Returns the center of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy)

voronoi_centers = []
voronoi_edges = []

edge_list_for_plotting = []
for vertex in dcel_vertex_list:
    if vertex != None:
        edge = vertex.inc_edge
        
        edge_for_plot1 = [(vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1]), (vertices[edge.twin.origin_vertex.index][0], vertices[edge.twin.origin_vertex.index][1])]
        edge_for_plot2 = [(vertices[edge.next_edge.origin_vertex.index][0], vertices[edge.next_edge.origin_vertex.index][1]), (vertices[edge.next_edge.twin.origin_vertex.index][0], vertices[edge.next_edge.twin.origin_vertex.index][1])]
        edge_for_plot3 = [(vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1]), (vertices[edge.prev_edge.twin.origin_vertex.index][0], vertices[edge.prev_edge.twin.origin_vertex.index][1])]
        edge_list_for_plotting = edge_list_for_plotting + [edge_for_plot1, edge_for_plot2, edge_for_plot3]
        
        center = define_center((vertices[edge.origin_vertex.index][0], vertices[edge.origin_vertex.index][1]),
        (vertices[edge.twin.origin_vertex.index][0], vertices[edge.twin.origin_vertex.index][1]),
        (vertices[edge.prev_edge.origin_vertex.index][0], vertices[edge.prev_edge.origin_vertex.index][1])
        )
        voronoi_centers = voronoi_centers + [center]
        # ax.scatter(center[0], center[1], c=color_for_plotting)

        
        # edge = edge.next_edge

color_for_plotting = (0.3, 0.3, 0.6, 1)
x_coords = [i[0] for i in vertices]
y_coords = [i[1] for i in vertices]


fig, ax = plt.subplots()
ax.scatter(x_coords, y_coords, c=color_for_plotting)
ax.scatter([i[0] for i in voronoi_centers], [i[1] for i in voronoi_centers])
for vertex_index in vertex_indices_with_neighbors:
    ax.annotate(vertex_index.vertex_index, (vertices[vertex_index.vertex_index][0], vertices[vertex_index.vertex_index][1]), xytext=(vertices[vertex_index.vertex_index][0] + 0.015, vertices[vertex_index.vertex_index][1]), c=color_for_plotting)
lc = mc.LineCollection(edge_list_for_plotting, colors=(0.0, 0.75, 0.0, 1),linewidths=2)
ax.add_collection(lc)
ax.autoscale()
plt.show()
