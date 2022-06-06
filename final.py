import numpy as np
from typing import Any
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from dataclasses import dataclass
from matplotlib import collections  as mc


#np.random.seed(2)
@dataclass
class Vertex:
    x: float
    y: float
    index: int

@dataclass
class Half_Edge:
    origin: Vertex
    target: Vertex


def plot_dcel_edge(dcel_edge_list):
  _, ax = plt.subplots()
  colors = iter(cm.rainbow(np.linspace(0, 1, len(vertexes))))
  edge_list = [[np.array([halfedge.origin.x,halfedge.origin.y]),np.array([halfedge.target.x,halfedge.target.y])] for halfedge in dcel_edge_list]
  lc = mc.LineCollection(edge_list, colors=(0.0, 0.75, 0.0, 1),linewidths=2, color=next(colors))
  ax.add_collection(lc)
  ax.autoscale()
  plt.show()


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


vertexes = np.loadtxt("ExampleInput.txt")
convex_hull = [[vertexes[i],vertexes[(i+1)%len(vertexes)]] for i in range(len(vertexes))]
vertexes_permutated = np.random.permutation(vertexes)
vertexes_permutated_index = [np.where(vertexes==i)[0][0] for i in vertexes_permutated]
# fix permutation for debug
#vertexes_permutated = [[0.9631,0.2619],[0.2316,0.1366],[0.988 ,0.715 ],[0.0987,0.1978],[0.8852,0.8909],[0.0377,0.9037],[0.6241,0.1068]]
#vertexes_permutated_index = [3, 1, 4, 0, 5, 6, 2]

# deleted permunated vertex one by one, and record their neighbours
neighbours_stack = []
neighbours_stack_id = []
vertexes_index = np.arange(len(vertexes))
for i in vertexes_permutated_index:
  i_index = np.where(vertexes_index==i)[0]
  neighbours_next = vertexes_index[np.asscalar((i_index+1)%len(vertexes_index))]
  neighbours_prev = vertexes_index[np.asscalar(i_index-1)]
  neighbours_stack.append([vertexes[neighbours_prev],vertexes[neighbours_next]])
  neighbours_stack_id.append([neighbours_prev,neighbours_next])
  vertexes_index = np.delete(vertexes_index,i_index)
  # print("current vertex",i, ",index in current convex hull is", i_index)
  # print("vertex",i,"remaining neighbours are ",neighbours_cw,neighbours_ccw)
  # print("deleted vertex",i,"from convex hull")
  # print("Remaining CH vertexes are:",vertexes_index)
  if len(vertexes_index) <=3:
    break
print(vertexes)
print(vertexes_permutated)
print(vertexes_permutated_index)
print(neighbours_stack)

#plot_edge(neighbours)
#plot_edge(convex_hull)

vertexes_permutated_reverse = vertexes_permutated[::-1]
neighbours_stack_reverse = neighbours_stack[::-1]
neighbours_stack_id_reverse = neighbours_stack_id[::-1]
vertexes_permutated_index_reverse = vertexes_permutated_index[::-1]
base_vertex_id = np.array(sorted(vertexes_permutated_index_reverse[:3]))

def swap_test(a: Vertex,b: Vertex,c: Vertex, vertexes_list_dcel:Vertex, edge_list_dcel:Half_Edge):
  if np.abs(b.index - c.index) == 1 or np.abs(b.index - c.index) == 6:
    print("boundary, return")
    return edge_list_dcel
  else:
    for d in vertexes_list_dcel:
      for edge1 in edge_list_dcel:
        for edge2 in edge_list_dcel:
          if edge1.origin == c and edge1.target == d and edge2.origin == d and edge2.target == b:
            print("a:", a.index,"b:", b.index,"c:", c.index,"d:", d.index)   
            if inCircleTest(b.x,b.y,c.x,c.y,d.x,d.y,a.x,a.y):
              print("flip bc:",b.index,c.index, "to ad:", a.index, d.index)
              edge_list_dcel.remove(Half_Edge(origin=b,target=c))
              edge_list_dcel.remove(Half_Edge(origin=c,target=b))
              edge_list_dcel.append(Half_Edge(origin=a,target=d))
              edge_list_dcel.append(Half_Edge(origin=d,target=a))
              print("propagate swap adc", a.index, d.index, c.index)
              edge_list_dcel = swap_test(a, d, c, vertexes_list_dcel,edge_list_dcel)
              print("propagate swap abd", a.index, b.index, d.index)
              edge_list_dcel = swap_test(a, b, d, vertexes_list_dcel,edge_list_dcel)
              return edge_list_dcel
            else:
              print("No need of flip ")
    #print("Intermediate boundary, return")
    return edge_list_dcel



vertexes_dcel = []
edge_list_dcel = []
vertexes_list_dcel = []
for vertex_id in range(len(vertexes)):
  vertexes_dcel.append(Vertex(x=vertexes[vertex_id][0],y=vertexes[vertex_id][1],index=vertex_id))
edge_list_dcel = [Half_Edge(origin=vertexes_dcel[vertex1],target = vertexes_dcel[vertrx2]) for vertex1, vertrx2 in zip(base_vertex_id,np.roll(base_vertex_id,-1))]
vertexes_list_dcel = [vertexes_dcel[i] for i in base_vertex_id]

for id in range(len(vertexes_permutated_index_reverse[3:])):
  add_vertex_dcel = vertexes_dcel[vertexes_permutated_index_reverse[id+3]]
  add_vertex_prev_dcel = vertexes_dcel[neighbours_stack_id_reverse[id][0]]
  add_vertex_next_dcel = vertexes_dcel[neighbours_stack_id_reverse[id][1]]
  edge_list_dcel.append(Half_Edge(origin=add_vertex_prev_dcel,target = add_vertex_dcel))
  edge_list_dcel.append(Half_Edge(origin=add_vertex_dcel,target = add_vertex_next_dcel))
  edge_list_dcel.append(Half_Edge(origin=add_vertex_next_dcel,target = add_vertex_prev_dcel))
  vertexes_list_dcel.append(add_vertex_dcel)
  print("insert point: ",add_vertex_dcel.index)
  edge_list_dcel = swap_test(add_vertex_dcel,add_vertex_prev_dcel,add_vertex_next_dcel,vertexes_list_dcel, edge_list_dcel)
  plot_dcel_edge(edge_list_dcel)

edge_list_dcel

  

