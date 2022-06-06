import numpy as np
import matplotlib.pyplot as plt



def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
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

faces = [[(0,0), (1,3), (2,-1)],[(1,3), (2,-1), (4,0)],[(1,3), (4,0), (6,7)]]
center = [(np.Inf, np.Inf)]
face_x = []
face_y = []
for i in faces:
    center.append(define_circle(i[0],i[1],i[2]))
    for j in range(3):
        face_x.append(i[j][0])
        face_y.append(i[j][1])
    face_x.append(i[0][0])
    face_y.append(i[0][1])
    
print(center)

x = []
y = []
voronoi_edge = []
for i in range(0, len(center)):
    x.append(center[i][0])
    y.append(center[i][1])
    for j in range(3):
        voronoi_edge.append(-(face_x[j+1+i]-face_x[j+i])/(face_y[j+1+i]-face_y[j+i]))
    for v in voronoi_edge:
        plt.axline(center[i], slope = v)
    
plt.scatter(x, y)
plt.plot(face_x, face_y)
plt.scatter(face_x, face_y)
# plt.axline(center[1], slope=-(face_x[1]-face_x[0])/(face_y[1]-face_y[0]))
plt.show()