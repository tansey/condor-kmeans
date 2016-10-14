import matplotlib.pylab as plt
import numpy as np

points = np.loadtxt('test/data.csv', delimiter=',')
centroids = np.loadtxt('test/centroids.csv', delimiter=',')
kmeans_assignments = np.loadtxt('test/assignments.csv', delimiter=',', dtype=int)

cluster_colors = ['blue', 'orange', 'purple', 'goldenrod', 'pink']
plt.scatter(points[:,0], points[:,1], alpha=0.5, color=[cluster_colors[i] for i in kmeans_assignments])
plt.scatter(centroids[:,0], centroids[:,1], color=cluster_colors, s=np.unique(kmeans_assignments, return_counts=True)[1])
plt.show()
