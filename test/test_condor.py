import sys
import os
import numpy as np
from condor_kmeans.condor import CondorKmeans

if __name__ == '__main__':
    num_points = 10000
    num_clusters = 5
    num_dimensions = 2
    cluster_colors = ['blue', 'orange', 'purple', 'goldenrod', 'pink']
    clusters = np.random.random(size=(num_clusters, num_dimensions)) * 20 - 10
    noise = np.random.normal(size=(num_points, num_dimensions))
    assignments = np.random.choice(np.arange(num_clusters), replace=True, size=num_points)
    points = np.ma.masked_array(clusters[assignments] + noise, mask=np.isnan(clusters[assignments] + noise))

    max_steps = 5
    num_threads = 3
    num_workers = 21 # odd number just to test the rounding
    username = sys.argv[1]
    pool = CondorKmeans(username, num_workers, os.getcwd(), 'data/test_centroids.csv', 'data/test_assignments.csv', max_steps=max_steps)
    pool.weighted_kmeans(points, np.ones(2), num_clusters, num_threads,
                                                         pp_init=False, pp_reservoir_size=1000, pp_max=3)

    np.savetxt('data/test_data.csv', points, delimiter=',')
    np.savetxt('data/test_true_clusters.csv', clusters, delimiter=',')
    np.savetxt('data/test_true_points.csv', assignments, delimiter=',')