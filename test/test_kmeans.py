import matplotlib.pylab as plt
import numpy as np
from condor_kmeans.kmeans import weighted_kmeans

if __name__ == '__main__':
    num_points = 1000
    num_clusters = 5
    num_dimensions = 2
    cluster_colors = ['blue', 'orange', 'purple', 'goldenrod', 'pink']
    clusters = np.random.random(size=(num_clusters, num_dimensions)) * 20 - 10
    noise = np.random.normal(size=(num_points, num_dimensions))
    assignments = np.random.choice(np.arange(num_clusters), replace=True, size=num_points)
    points = np.ma.masked_array(clusters[assignments] + noise, mask=np.isnan(clusters[assignments] + noise))

    max_steps = 50
    num_threads = 3
    kmeans_assignments, centroids = weighted_kmeans(points, np.ones(2), num_clusters, max_steps, num_threads, pp_init=False, pp_reservoir_size=1000, pp_max=100)
    
    plt.scatter(points[:,0], points[:,1], alpha=0.5, color=[cluster_colors[i] for i in kmeans_assignments])
    plt.scatter(clusters[:,0], clusters[:,1], color='red', s=np.unique(assignments, return_counts=True)[1])
    plt.scatter(centroids[:,0], centroids[:,1], color=cluster_colors, s=np.unique(kmeans_assignments, return_counts=True)[1])
    plt.show()
