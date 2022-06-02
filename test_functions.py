import numpy as np

array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(array[:-3])
print(array[-3:])

array = array + [2,4]
print(array)

matrix = np.array([[1,3,2,0],
          [3,0,9,8],
          [4,7,1,0],
          [0,0,0,1]])
det = np.linalg.det(matrix)
print(det)

array = []
array.append