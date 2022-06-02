from copy import deepcopy
from operator import indexOf
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from dataclasses import dataclass
from copy import deepcopy
from typing import Any

#np.random.seed(2)
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

## Read the input file. Vertices are already in convex position
vertices = np.loadtxt("ExampleInput.txt", dtype=np.float32)
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

initial_edge_list = []

dcel_vertex_list = [None] * len(vertices)

## Get the first vertex that we're considering. From it, we will have all the necessary information to build the initial triangle
vertex_index_with_neighbor = vertex_indices_with_neighbors[-1]

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

## We now have the initial triangle. We start with adding the other vertices 
# one by one and doing the inCircle Test, before deciding if we flip or not.

## d is the new point
## a is d.next (neighbor)
## b is a.next (next vertex around the incident face)
## c is d.prev == b.next == a.next.next
def inCircleTest(a_x: float, a_y: float,
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

## We start inserting points in reverse order of deletion
for vertex_index_with_neighbor in reversed(vertex_indices_with_neighbors[:-3]):
    
    ## Get the indices

    cur_vertex_index = vertex_index_with_neighbor.vertex_index
    next_vertex_index = vertex_index_with_neighbor.next_neighbor_index
    prev_vertex_index = vertex_index_with_neighbor.prev_neighbor_index
    third_point_on_triangle = dcel_vertex_list[next_vertex_index].inc_edge.twin.origin_vertex.index

    ##Plot the area we're working in
    # plt.plot([vertices[next_vertex_index][0], vertices[prev_vertex_index][0], vertices[third_point_on_triangle][0], vertices[next_vertex_index][0]], [vertices[next_vertex_index][1], vertices[prev_vertex_index][1], vertices[third_point_on_triangle][1], vertices[next_vertex_index][1]])
    # plt.scatter([vertices[cur_vertex_index][0]], [vertices[cur_vertex_index][1]])
    # plt.show()

    ## Create the new vertex
    dcel_vertex_list[cur_vertex_index] = Vertex(inc_edge=None, index=cur_vertex_index, x=vertices[cur_vertex_index][0], y=vertices[cur_vertex_index][1])

    ## Check if we need to perform a flip
    needToFlip = inCircleTest(d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                 a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                 b_x=vertices[third_point_on_triangle][0], b_y=vertices[third_point_on_triangle][1],
                 c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1])
    #print(f"Need to Flip: {needToFlip}")

    ## In case the point is inside the circle, we do the necessary updates, and propagate.
    if needToFlip:
        while(needToFlip):
            ## Delete the previous edge
            #dcel_vertex_list[prev_vertex_index].inc_edge = None

            new_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[next_vertex_index].inc_edge.next_edge, prev_edge=None, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=dcel_vertex_list[next_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[third_point_on_triangle], twin=None))
            new_edge.twin.twin = new_edge

            dcel_vertex_list[next_vertex_index].inc_edge.prev_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[next_vertex_index].inc_edge, prev_edge=new_edge.twin, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None,next_edge=None,prev_edge=None,origin_vertex=dcel_vertex_list[next_vertex_index],twin=None))
            dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.twin.twin = dcel_vertex_list[next_vertex_index].inc_edge.prev_edge

            dcel_vertex_list[prev_vertex_index].inc_edge.twin.origin_vertex = dcel_vertex_list[cur_vertex_index]
            dcel_vertex_list[prev_vertex_index].inc_edge.next_edge = new_edge

            new_edge.prev_edge = dcel_vertex_list[prev_vertex_index].inc_edge

            new_edge.twin.next_edge = dcel_vertex_list[next_vertex_index].inc_edge.prev_edge
            dcel_vertex_list[next_vertex_index].inc_edge.next_edge = new_edge.twin            

            dcel_vertex_list[cur_vertex_index].inc_edge = dcel_vertex_list[next_vertex_index].inc_edge.prev_edge
            
            face1 = Face(inc_edge=new_edge)
            face2 = Face(inc_edge=new_edge.twin)

            new_edge.inc_face = face1
            new_edge.prev_edge.inc_face = face1
            new_edge.next_edge.inc_face = face1

            new_edge.twin.inc_face = face2
            new_edge.twin.prev_edge.inc_face = face2
            new_edge.twin.next_edge.inc_face = face2

            # Debugging
            print(f"\nFlip?: {needToFlip}")
            edge1 = face1.inc_edge
            edge2 = face2.inc_edge
            counter = 0
            while counter < 10:
                counter += 1
                print(f"vertex1: {edge1.origin_vertex.index}")
                edge1 = edge1.next_edge
            counter = 0
            print()
            while counter < 10:
                counter += 1
                print(f"vertex2: {edge2.origin_vertex.index}")
                edge2 = edge2.next_edge
            next_vertex_index = third_point_on_triangle
            
            # if not dcel_vertex_list[third_point_on_triangle].inc_edge.twin.next_edge: #if we don't have a next edge for the twin, then we're on the outer hull
            #     needToFlip = False
            if False:
                print("whatever")
            else:
                #third_point_on_triangle = dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.next_edge.twin.origin_vertex.index
                third_point_on_triangle = dcel_vertex_list[next_vertex_index].inc_edge.twin.origin_vertex.index
                needToFlip = inCircleTest(
                d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                b_x=vertices[third_point_on_triangle][0], b_y=vertices[third_point_on_triangle][1],
                c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1]
                )

    else:
        # prev_inc = dcel_vertex_list[prev_vertex_index].inc_edge
        # prev_twin = prev_inc.twin

        # dcel_vertex_list[next_vertex_index].inc_edge = prev_twin
        # dcel_vertex_list[next_vertex_index].inc_edge.twin = prev_inc

        dcel_vertex_list[cur_vertex_index].inc_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[prev_vertex_index].inc_edge.twin, prev_edge=None, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=None, origin_vertex=dcel_vertex_list[next_vertex_index], twin=None))
        dcel_vertex_list[cur_vertex_index].inc_edge.twin.twin = dcel_vertex_list[cur_vertex_index].inc_edge

        #print(f"curvertex1: {dcel_vertex_list[cur_vertex_index].inc_edge.origin_vertex.index}")

        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge = Half_Edge(inc_face=None, prev_edge=dcel_vertex_list[cur_vertex_index].inc_edge.next_edge, next_edge=dcel_vertex_list[cur_vertex_index].inc_edge, origin_vertex=dcel_vertex_list[prev_vertex_index], twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=None, origin_vertex=dcel_vertex_list[cur_vertex_index], twin=None))
        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge.twin.twin = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge

        dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.next_edge = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge

        #print(f"prevvertex1: {dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge.origin_vertex.index}")

        dcel_vertex_list[prev_vertex_index].inc_edge = dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge
        
        #print(f"curvertex2: {dcel_vertex_list[prev_vertex_index].inc_edge.twin.origin_vertex.index}")

        face = Face(inc_edge=dcel_vertex_list[cur_vertex_index].inc_edge)

        dcel_vertex_list[cur_vertex_index].inc_edge.inc_face = face
        dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.inc_face = face
        dcel_vertex_list[cur_vertex_index].inc_edge.prev_edge.inc_face = face

        # print(f"face inc: {face.inc_edge.origin_vertex.index}")
        # print(f"face inc next: {face.inc_edge.next_edge.origin_vertex.index}")
        # print(f"face inc prev: {face.inc_edge.prev_edge.origin_vertex.index}")

        # Debugging
        print(f"\nFlip?: {needToFlip}")
        edge1 = face.inc_edge
        edge2 = dcel_vertex_list[cur_vertex_index].inc_edge.next_edge.twin.inc_face.inc_edge
        counter = 0
        while counter < 10:
            counter += 1
            print(f"vertex1: {edge1.origin_vertex.index}")
            edge1 = edge1.next_edge
        counter = 0
        print()
        while counter < 10:
            counter += 1
            print(f"vertex2: {edge2.origin_vertex.index}")
            edge2 = edge2.next_edge

edge_list_for_plotting = []
for vertex in dcel_vertex_list:
        
    #print(f"Vertex: {vertex.index}")
# print(f"Next: {vertex.inc_edge}")
#print(f"Prev: {vertex.inc_edge.prev_edge.origin_vertex.index}")

    edge1 = [(vertex.x, vertex.y), (vertex.inc_edge.next_edge.origin_vertex.x, (vertex.inc_edge.next_edge.origin_vertex.y))]
    edge2 = [(vertex.x, vertex.y), (vertex.inc_edge.prev_edge.origin_vertex.x, (vertex.inc_edge.prev_edge.origin_vertex.y))]
    edge3 = [(vertex.inc_edge.next_edge.origin_vertex.x, (vertex.inc_edge.next_edge.origin_vertex.y)), (vertex.inc_edge.prev_edge.origin_vertex.x, (vertex.inc_edge.prev_edge.origin_vertex.y))]
    edge_list_for_plotting = edge_list_for_plotting + [edge1, edge2, edge3]
    fig, ax = plt.subplots()
#lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    lc = mc.LineCollection(edge_list_for_plotting, colors=(0.0, 0.75, 0.0, 1),linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    plt.show()

# Debugging
for vertex in dcel_vertex_list:
    edge = vertex.inc_edge
    print(f"vertex index: {vertex.index}")
    counter = 0
    while counter < 10:
        counter += 1
        print(f"vertex: {edge.origin_vertex.index}")
        edge = edge.next_edge
    print()


