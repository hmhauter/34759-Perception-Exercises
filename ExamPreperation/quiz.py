import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression



# w = np.array([[0,0,0],[0,1,0],[0,0,0]])
# v = (1/9) * np.ones((3,3))
# a = 0.9
# f = ((1+a)*w-a*v)
# print(f)


# P = np.array([[725,0,631],[0,726,360],[0,0,1]])
# X = [1,1,4]
# x = (P @ X)
# x = x / x[2]
# print(x)

# A = np.array([[10,15,20],[20,20,25],[10,15,20]])
# B = np.array([[15,15,15],[2812.250,20,20],[30,30,30]])

# print(A-B)
# print((A-B)**2)
# print(np.sum((A-B)**2))
# print(np.sqrt(np.sum((A-B)**2)))

# x = np.loadtxt('/home/hhauter/Documents/S23/Perception/34759-Perception-Exercises/ExamPreperation/cluster.txt')
# print(x)
# wcss = []

# # Compute the within-cluster sum of squares for values of k from 1 to 10
# for k in range(1, 15):
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)

# # Plot the within-cluster sum of squares against the number of clusters
# plt.plot(range(1, 15), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Within-Cluster Sum of Squares')
# plt.show()


# x = np.loadtxt('/home/hhauter/Documents/S23/Perception/34759-Perception-Exercises/ExamPreperation/data1.txt')
# y = np.loadtxt('/home/hhauter/Documents/S23/Perception/34759-Perception-Exercises/ExamPreperation/data2.txt')


# x_r = x.reshape(-1,1) #fit needs x in this shape
# # Initialise and fit model
# lm = LinearRegression()
# model = lm.fit(x_r, y)

# predicted = model.predict(np.array([1]).reshape(-1, 1))
# print(predicted)
import cv2
rvec = np.array([-0.05, -1.51, -0.00])
tvec = np.array([87.39, -2.25, -24.89])
rmat, _ = cv2.Rodrigues(rvec)
print(rmat)

p_img = np.array([-6.71, 0.23, 21.59,1])

P = np.array([[ 0.0609624 ,  0.03109396 ,-0.99765563, 87.39],
 [ 0.03109396 , 0.9989704  , 0.03303495, -2.25],
 [ 0.99765563 ,-0.03303495 , 0.0599328 , -24.89],
 [0,0,0,1]])

print(np.linalg.pinv(P) @ p_img)





