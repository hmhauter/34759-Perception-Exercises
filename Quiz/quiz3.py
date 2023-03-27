# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Load the data from file
# X = np.loadtxt('Quiz/clusters.txt')

# # Initialize a list to store the within-cluster sum of squares for different values of k
# wcss = []

# # Compute the within-cluster sum of squares for values of k from 1 to 10
# for k in range(1, 15):
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # Plot the within-cluster sum of squares against the number of clusters
# plt.plot(range(1, 15), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Within-Cluster Sum of Squares')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from files
x = np.loadtxt('Quiz/data1.txt')
y = np.loadtxt('Quiz/data2.txt')

# Reshape the arrays to ensure they are 2D
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create a linear regression model and fit it to the data
model = LinearRegression().fit(x, y)

# Extract the slope and intercept from the model
a = model.coef_[0][0]
b = model.intercept_[0]

# Print the values of a and b
print("a =", a)
print("b =", b)