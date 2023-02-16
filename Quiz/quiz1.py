import numpy as np
w = np.array([[0,0,0],[0,1,0],[0,0,0]])
v = np.array([[1,1,1],[1,1,1], [1,1,1]]) * (1/9)

alpha = 0.9

kernel = (1+alpha) * w - alpha * v

print(kernel)

