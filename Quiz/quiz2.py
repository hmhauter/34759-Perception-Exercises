import numpy as np 
def calculate_sum_differences(img1, img2):

    return np.sum(np.abs(img1-img2)**2)


A = [[10,15,20], [20,20,25],[10,15,20]]
A = np.array(A)

B = [[15,15,15], [20,20,20], [30,30,30]]
B = np.array(B)

print(A)
print(B)

res = calculate_sum_differences(A, B)
print(res)

C = np.eye(3,3)
print(C)
D = [[725,0,631],[0,726,360],[0,0,1]]
D = np.array(D)
p = [1,1,3]
p = np.array(p)
p_cam = D @ p
print(p_cam)
print(p_cam / 3.0)


