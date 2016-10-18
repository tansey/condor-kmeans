import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

if __name__ == '__main__':
    points = np.loadtxt('data/test_data.csv', delimiter=',')
    clusters = np.loadtxt('data/test_truth.csv', delimiter=',')
    centroids = np.loadtxt('data/test_centroids.csv', delimiter=',')
    kmeans_assignments = np.loadtxt('data/test_assignments.csv', delimiter=',', dtype=int)
    plt.scatter(points[:,0], points[:,1], alpha=0.5, color=[cluster_colors[i] for i in kmeans_assignments])
    plt.scatter(clusters[:,0], clusters[:,1], color='red', s=np.unique(assignments, return_counts=True)[1])
    plt.scatter(centroids[:,0], centroids[:,1], color=cluster_colors, s=np.unique(kmeans_assignments, return_counts=True)[1])
    plt.savefig('data/test_condor.pdf', bbox_inches='tight')