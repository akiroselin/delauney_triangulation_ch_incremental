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
    plt.plot([vertices[next_vertex_index][0], vertices[prev_vertex_index][0], vertices[third_point_on_triangle][0], vertices[next_vertex_index][0]], [vertices[next_vertex_index][1], vertices[prev_vertex_index][1], vertices[third_point_on_triangle][1], vertices[next_vertex_index][1]])
    plt.scatter([vertices[cur_vertex_index][0]], [vertices[cur_vertex_index][1]])
    plt.show()

    ## Create the new vertex
    cur_vertex = Vertex(inc_edge=None, index=cur_vertex_index, x=vertices[cur_vertex_index][0], y=vertices[cur_vertex_index][1])
    dcel_vertex_list[cur_vertex_index] = cur_vertex
    needToFlip = inCircleTest(d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                 a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                 b_x=vertices[third_point_on_triangle][0], b_y=vertices[third_point_on_triangle][1],
                 c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1])

    if needToFlip:
        #print(f"We're flipping index {cur_vertex_index}")
        # ## We do the necessary updates
        # print(f"Third next index: {dcel_vertex_list[third_point_on_triangle].inc_edge.next_edge.origin_vertex.index}")
        # print(f"Third prev index: {dcel_vertex_list[third_point_on_triangle].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"next next index: {dcel_vertex_list[next_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"next prev index: {dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"prev next index: {dcel_vertex_list[prev_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"prev prev index: {dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
        while(needToFlip):
            cur_vertex.inc_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[third_point_on_triangle].inc_edge,    prev_edge=dcel_vertex_list[prev_vertex_index].inc_edge, origin_vertex=cur_vertex, twin=Half_Edge   (inc_face=None, next_edge=None, prev_edge=dcel_vertex_list[next_vertex_index].inc_edge,    origin_vertex=dcel_vertex_list[third_point_on_triangle], twin=None))
            cur_vertex.inc_edge.twin.twin = cur_vertex.inc_edge

            dcel_vertex_list[prev_vertex_index].inc_edge.twin.origin_vertex = cur_vertex
            dcel_vertex_list[prev_vertex_index].inc_edge.next_edge = cur_vertex.inc_edge

            dcel_vertex_list[next_vertex_index].inc_edge.prev_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[next_vertex_index].inc_edge, prev_edge=cur_vertex.inc_edge.twin, origin_vertex=cur_vertex, twin=Half_Edge(inc_face=None, next_edge=None, prev_edge=None, origin_vertex=dcel_vertex_list[next_vertex_index], twin=None))
            dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.twin.twin = dcel_vertex_list[next_vertex_index].inc_edge.prev_edge

            face1 = Face(inc_edge=cur_vertex.inc_edge)
            face2 = Face(inc_edge=cur_vertex.inc_edge.twin)

            cur_vertex.inc_edge.inc_face = face1
            cur_vertex.inc_edge.prev_edge.inc_face = face1
            cur_vertex.inc_edge.next_edge.inc_face = face1

            cur_vertex.inc_edge.twin.inc_face = face2
            cur_vertex.inc_edge.twin.prev_edge.inc_face = face2
            cur_vertex.inc_edge.twin.next_edge = dcel_vertex_list[next_vertex_index].inc_edge.prev_edge
            cur_vertex.inc_edge.twin.next_edge.inc_face = face2
            
            next_vertex_index = third_point_on_triangle
            if not dcel_vertex_list[third_point_on_triangle].inc_edge.twin.next_edge:
                needToFlip = False
            else:
                third_point_on_triangle = dcel_vertex_list[third_point_on_triangle].inc_edge.twin.next_edge.twin.origin_vertex.index

                needToFlip = inCircleTest(
                d_x=vertices[cur_vertex_index][0], d_y=vertices[cur_vertex_index][1],
                a_x=vertices[next_vertex_index][0], a_y=vertices[next_vertex_index][1],
                b_x=vertices[third_point_on_triangle][0], b_y=vertices[third_point_on_triangle][1],
                c_x=vertices[prev_vertex_index][0], c_y=vertices[prev_vertex_index][1]
                )


    else:
        #print(f"We're NOT flipping index {cur_vertex_index}")
        # print(f"Third next index: {dcel_vertex_list[third_point_on_triangle].inc_edge.next_edge.origin_vertex.index}")
        # print(f"Third prev index: {dcel_vertex_list[third_point_on_triangle].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"next next index: {dcel_vertex_list[next_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"next prev index: {dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"prev next index: {dcel_vertex_list[prev_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"prev prev index: {dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
        prev_inc = dcel_vertex_list[prev_vertex_index].inc_edge
        prev_twin = prev_inc.twin

        dcel_vertex_list[next_vertex_index].inc_edge = prev_twin
        dcel_vertex_list[next_vertex_index].inc_edge.twin = prev_inc

        cur_vertex.inc_edge = Half_Edge(inc_face=None, next_edge=dcel_vertex_list[next_vertex_index].inc_edge,prev_edge=None, origin_vertex=cur_vertex, twin=Half_Edge(inc_face=None, origin_vertex=dcel_vertex_list[next_vertex_index], prev_edge=None, next_edge=None, twin=None))
        cur_vertex.inc_edge.twin.twin = cur_vertex.inc_edge

        cur_vertex.inc_edge.prev_edge = Half_Edge(inc_face=None, origin_vertex=dcel_vertex_list[prev_vertex_index], next_edge=cur_vertex.inc_edge, prev_edge=dcel_vertex_list[next_vertex_index].inc_edge, twin=Half_Edge(inc_face=None, origin_vertex=cur_vertex, prev_edge=None, next_edge=None, twin=None))
        cur_vertex.inc_edge.prev_edge.twin.twin = cur_vertex.inc_edge.prev_edge

        cur_vertex.inc_edge.next_edge.next_edge = cur_vertex.inc_edge.prev_edge

        dcel_vertex_list[next_vertex_index].inc_edge.next_edge = cur_vertex.inc_edge.prev_edge
        dcel_vertex_list[next_vertex_index].inc_edge.prev_edge = cur_vertex.inc_edge

        face = Face(inc_edge=cur_vertex.inc_edge)
        cur_vertex.inc_edge.inc_face = face
        cur_vertex.inc_edge.next_edge.inc_face = face
        cur_vertex.inc_edge.prev_edge.inc_face = face
        # print(f"Third next index: {dcel_vertex_list[third_point_on_triangle].inc_edge.next_edge.origin_vertex.index}")
        # print(f"Third prev index: {dcel_vertex_list[third_point_on_triangle].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"next index: {next_vertex_index}")
        # print(f"next next index: {dcel_vertex_list[next_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"next prev index: {dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
        # print(f"prev next index: {dcel_vertex_list[prev_vertex_index].inc_edge.next_edge.origin_vertex.index}")
        # print(f"prev prev index: {dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
    # print(f"Third next index: {dcel_vertex_list[third_point_on_triangle].inc_edge.next_edge.origin_vertex.index}")
    # print(f"Third prev index: {dcel_vertex_list[third_point_on_triangle].inc_edge.prev_edge.origin_vertex.index}")
    # print(f"next next index: {dcel_vertex_list[next_vertex_index].inc_edge.next_edge.origin_vertex.index}")
    # print(f"next prev index: {dcel_vertex_list[next_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
    # print(f"prev next index: {dcel_vertex_list[prev_vertex_index].inc_edge.next_edge.origin_vertex.index}")
    # print(f"prev prev index: {dcel_vertex_list[prev_vertex_index].inc_edge.prev_edge.origin_vertex.index}")
    # counter = 0
    # while counter < 10:
    #     print(f"cur vertex check: {cur_vertex.inc_edge.origin_vertex.index}")
    #     cur_vertex = cur_vertex.inc_edge.next_edge.origin_vertex
    #     counter += 1


edge_list_for_plotting = []
for vertex in dcel_vertex_list:
        
    #print(f"Vertex: {vertex.index}")
# print(f"Next: {vertex.inc_edge}")
#print(f"Prev: {vertex.inc_edge.prev_edge.origin_vertex.index}")

    edge1 = [(vertex.x, vertex.y), (vertex.inc_edge.next_edge.origin_vertex.x, (vertex.inc_edge.next_edge.origin_vertex.y))]
    edge2 = [(vertex.x, vertex.y), (vertex.inc_edge.prev_edge.origin_vertex.x, (vertex.inc_edge.prev_edge.origin_vertex.y))]
    edge3 = [(vertex.inc_edge.next_edge.origin_vertex.x, (vertex.inc_edge.next_edge.origin_vertex.y)), (vertex.inc_edge.prev_edge.origin_vertex.x, (vertex.inc_edge.prev_edge.origin_vertex.y))]
    edge_list_for_plotting = edge_list_for_plotting + [edge1, edge2, edge3]


# print(edge_list_for_plotting)
fig, ax = plt.subplots()
#lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
lc = mc.LineCollection(edge_list_for_plotting, colors=(0.0, 0.75, 0.0, 1),linewidths=2)
#lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
ax.add_collection(lc)
#ax.add_collection(lc2)
ax.autoscale()
plt.show()

# results = [0 if i == None else 1 for i in dcel_vertex_list]
# print(results)
# print(dcel_vertex_list[6].inc_edge.next_edge)
## Plot
# edge_list_for_plotting = []

##
#counter = 0

## Store the last vertex, which is the first one to be considered. It will be used to 
# make an edge with the next vertex
# prev_vertex_index = vertex_indices_with_neighbors[-1].vertex_index

# ## In the DCEL list, we're adding the vertex as defined at the exact index location, therefore we can access it immediately.
# dcel_vertex_list[prev_vertex_index] = Vertex(x=vertices[prev_vertex_index][0], y=vertices[prev_vertex_index][1],index=prev_vertex_index, inc_edge=None)

#store them in the edge list
#dcel_edge_list[0] = Half_Edge(inc_face=None, next_edge=None, prev_edge=None, twin=None, origin_vertex=dcel_vertex_list[prev_vertex_index])
#dcel_vertex_list[prev_vertex_index].inc_edge = dcel_edge_list[0]

# counter = 0
# ## Create the initial Triangle. We take the 2 points from the end of the array
# for vertex_index_with_neighbor in reversed(vertex_indices_with_neighbors[-3:]):
#     ## Get the indices
#     cur_vertex_index = vertex_index_with_neighbor.vertex_index
#     next_vertex_index = vertex_index_with_neighbor.next_neighbor_index
#     prev_vertex_index = vertex_index_with_neighbor.prev_neighbor_index

#     ## Create the vertices with the info currently available.
#     if dcel_vertex_list[cur_vertex_index] == None:
#         dcel_vertex_list[cur_vertex_index] = Vertex(x=vertices[cur_vertex_index][0], y=vertices[cur_vertex_index][1], index=cur_vertex_index, inc_edge=None)
#     if dcel_vertex_list[next_vertex_index] == None:
#         dcel_vertex_list[next_vertex_index] = Vertex(x=vertices[next_vertex_index][0], y=vertices[next_vertex_index][1], index=next_vertex_index, inc_edge=None)
#     if dcel_vertex_list[prev_vertex_index] == None:
#         dcel_vertex_list[prev_vertex_index] = Vertex(x=vertices[prev_vertex_index][0], y=vertices[prev_vertex_index][1], index=prev_vertex_index, inc_edge=None)
    
#     cur_inc_edge = Half_Edge(inc_face=None, next_edge=None, prev_edge=None, 
#     origin_vertex=dcel_vertex_list[cur_vertex_index], twin=Half_Edge(inc_face=None, 
#     next_edge=None,origin_vertex=dcel_vertex_list[next_vertex_index], prev_edge=None, twin=cur_inc_edge))
    
#     prev_inc_edge = Half_Edge(inc_face=None, next_edge=cur_inc_edge, prev_edge=None, 
#     origin_vertex=dcel_vertex_list[prev_vertex_index], twin=Half_Edge(inc_face=None, 
#     next_edge=None,origin_vertex=dcel_vertex_list[cur_vertex_index], prev_edge=None, twin=cur_inc_edge.twin))



    # dcel_edge_list[counter] = Half_Edge(inc_face=None, origin_vertex=dcel_vertex_list[cur_vertex_index], next_edge=None, 
    # prev_edge=None, twin=Half_Edge(inc_face=None, origin_vertex=dcel_vertex_list[cur_vertex_index], next_edge=None, 
    # prev_edge=None, twin=dcel_edge_list[counter]))
    # next_edge = Half_Edge(inc_face=None,origin_vertex=dcel_vertex_list[cur_vertex_index], next_edge=)
## We now have the 3 initial vertices
    #print(f"vertex_with_neighbor: {vertex_with_neighbor}")
    # edge = (prev_vertex_index, cur_vertex_index)
    #print(f"edge: {edge}")
    # initial_edge_list = initial_edge_list + [edge]
    #print(f"final_edge_list: {final_edge_list}")
    # prev_vertex_index = cur_vertex_index
    #fig, ax = plt.subplots()
    #edges_for_plotting = []
    #edges_for_plotting = [[(i.start_vertex.x, i.start_vertex.y), (i.end_vertex.x, i.end_vertex.y)] for i in final_edge_list]
    #lc = mc.LineCollection(edges_for_plotting, colors=(0.3, 0.3, 0.6, 1), linewidths=2)
    #ax.add_collection(lc)
    #ax.autoscale()
    #plt.show()
    #lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    #lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
    #ax.add_collection(lc2)
    
    #counter += 1
    #if counter == 2:
        # edge = Edge(cur_vertex, vertices_with_neighbors[-1].vertex)
        # final_edge_list = final_edge_list + [edge]

## This is to close our triangle
# edge = (cur_vertex_index, vertex_indices_with_neighbors[-1].vertex_index)
# initial_edge_list = initial_edge_list + [edge]

# ## Now inside initial_edge_list we have the initial triangle. 
# # We will create a DCEL starting from that.
# for vertex in vertices:
#     vertex = Vertex(edge[0])


# ## Plot Initial Triangle
# fig, ax = plt.subplots()
# edges_for_plotting = []
# edges_for_plotting = [[(i.start_vertex.x, i.start_vertex.y), (i.end_vertex.x, i.end_vertex.y)] for i in final_edge_list]
# color_for_plotting = (0.3, 0.3, 0.6, 1)
# lc = mc.LineCollection(edges_for_plotting, colors=color_for_plotting, linewidths=2)
# ax.add_collection(lc)
# #lines2 = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
# #lc2 = mc.LineCollection(lines2, colors=(0.0, 0.2, 0.4, 1), linewidths=2)
# #ax.add_collection(lc2)
# #ax.annotate(str(edge.start_vertex.index), (edge.start_vertex.x, edge.start_vertex.y))
# print(f"vertices_with_neighbors: {vertices_with_neighbors}")
# x_coords = [i[0] for i in vertices]
# y_coords = [i[1] for i in vertices]
# ax.scatter(x_coords, y_coords, c=color_for_plotting)
# for vertex in vertices_with_neighbors:
#     ax.annotate(vertex.vertex.index, (vertex.vertex.x, vertex.vertex.y), xytext=(vertex.vertex.x + 0.015, vertex.vertex.y), c=color_for_plotting)
# ax.autoscale()

# plt.show()

# new_vertex_with_neighbor = vertices_with_neighbors[-4]
# new_edge_1 = Edge(new_vertex_with_neighbor.vertex, Vertex(vertices[new_vertex_with_neighbor.prev_neighbor_index][0], vertices[new_vertex_with_neighbor.prev_neighbor_index][1], new_vertex_with_neighbor.prev_neighbor_index))
# new_edge_2 = Edge(new_vertex_with_neighbor.vertex, Vertex(vertices[new_vertex_with_neighbor.next_neighbor_index][0], vertices[new_vertex_with_neighbor.next_neighbor_index][1], next_neighbor_index))


# final_edge_list = final_edge_list + [new_edge_1, new_edge_2]
# fig, ax = plt.subplots()
# edges_for_plotting = []
# edges_for_plotting = [[(i.start_vertex.x, i.start_vertex.y), (i.end_vertex.x, i.end_vertex.y)] for i in final_edge_list]
# color_for_plotting = (0.3, 0.3, 0.6, 1)
# lc = mc.LineCollection(edges_for_plotting, colors=color_for_plotting, linewidths=2)
# ax.add_collection(lc)
# x_coords = [i[0] for i in vertices]
# y_coords = [i[1] for i in vertices]
# ax.scatter(x_coords, y_coords, c=color_for_plotting)
# for vertex in vertices_with_neighbors:
#     ax.annotate(vertex.vertex.index, (vertex.vertex.x, vertex.vertex.y), xytext=(vertex.vertex.x + 0.015, vertex.vertex.y), c=color_for_plotting)
# ax.autoscale()

# plt.show()
## Add vertices one by one, and perform inCircle tests and flip edges accordingly

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
