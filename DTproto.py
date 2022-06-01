import numpy as np

vertexes = np.loadtxt("ExampleInput.txt", dtype=np.float32)
vertexes_permutated = np.random.permutation(vertexes)
print(vertexes)
print(vertexes_permutated)
vertexes_permutated_index = [np.where(vertexes==i)[0][0] for i in vertexes_permutated]

# deleted permunated vertex one by one, and record their neighbours
neighbours_with_deletion = []
vertexes_index = np.arange(len(vertexes))
for i in vertexes_permutated_index:
  i_index = np.where(vertexes_index==i)[0]
  neighbours_ccw = vertexes_index[np.asscalar((i_index+1)%len(vertexes_index))]
  neighbours_cw = vertexes_index[np.asscalar(i_index-1)]
  neighbours_with_deletion.append([vertexes[neighbours_cw],vertexes[neighbours_ccw]])
  vertexes_index = np.delete(vertexes_index,i_index)
  print("current vertex",i, ",index in current convex hull is", i_index)
  print("vertex",i,"remaining neighbours are ",neighbours_cw,neighbours_ccw)
  print("deleted vertex",i,"from convex hull")
  print("Remaining CH vertexes are:",vertexes_index)
  if len(vertexes_index) <=2:
    break
print(neighbours_with_deletion)

#
