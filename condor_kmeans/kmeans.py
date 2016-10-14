'''
Clusters foods using a weighted, probabilistic K-means.

Each point to cluster is a food in vector form:

    - Word2vec with 20 dimensions for the description

    - Nutrients on a per-calorie basis

The known values are all standardized.

For missing nutrient information, the dimension is skipped when calculating
distance and centroids.

The word2vec dimensions have a single weight associated with them.
'''
import sys
import numpy as np
import numpy.ma as ma
import collections
import argparse
import csv
from multiprocessing import Pool
from condor_kmeans.vector import VectorStream
from condor_kmeans.condor import condor_find_nearest_cluster

def calc_distances(x, centroids, weights):
    '''
    Calculates the distance from the data point x to each of the centroids. If
    x has a missing dimension, that dimension is not considered in the distance
    calculation. Uses cosine similarity for distance.
    '''
    return -(
            (x * centroids * weights).sum(axis=1)
            / (np.sqrt((x*x*weights).sum())
                * np.sqrt((centroids*centroids*weights).sum(axis=1)))
            )

def find_nearest_cluster(all_params):
    # All parameters are passed as a single tuple for use with thread pooling
    step, data, start, end, weights, centroids, assignments, thread_id = all_params

    min_distances = np.zeros(assignments.shape)

    # Iterate over every data point and (potentially) assign it to a new
    # cluster.
    for i, x in enumerate(data):
        # Easiest way to handle streaming and batch case
        if i < start:
            continue
        if i >= end:
            break

        # If we are on the initial step and this data point was chosen as
        # an initial centroid, just skip it.
        if step == 0 and assignments[i-start] != -1:
            continue

        # Find the distance from this data point to the cluster centroids.
        distances = calc_distances(x, centroids, weights)
        
        # If the entire vector was missing, arbitrarily assign it to the first cluster
        if distances.mask.min():
            distances[0] = 0
        
        # Find the minimum distance
        min_distances[i-start] = np.ma.min(distances)

        # Choose the nearest cluster.
        assignments[i-start] = int(np.argmin(distances))
        
    return (thread_id, assignments, min_distances)

def parallel_find_nearest_cluster(data, weights, centroids, assignments, num_threads, step, assign=True):
    min_distances = np.zeros(assignments.shape)

    # Create the thread pool to process things in parallel
    pool = Pool(num_threads)
    
    # Figure out how many data points each thread should be working on
    thread_load = len(data) / num_threads

    # Each thread is given a range in the form (inclusive_start, exclusive_end)
    thread_ranges = [(i*thread_load, (i+1)*thread_load) for i in xrange(num_threads)]
    thread_ranges[-1] = (thread_ranges[-1][0], len(data))

    # Create the parameters for each worker thread
    thread_params = [(step, data, start, end, weights, centroids, assignments[start:end], i) for i, (start, end) in enumerate(thread_ranges)]
    
    # Perform the K-Means iteration in parallel
    thread_results = pool.map(find_nearest_cluster, thread_params)

    # Update the assignments using the worker results
    for thread_id, thread_assignments, thread_min_distances in thread_results:
        start, end = thread_ranges[thread_id]
        min_distances[start:end] = thread_min_distances
        if assign:
            assignments[start:end] = thread_assignments

    # Kill the worker threads
    pool.terminate()

    return min_distances

def down_sample(data, reservoir_size):
    '''Down-sample our data for efficiency, while still preserving the distribution (note: variance may increase).'''
    reservoir = []
    reservoir_ids = []

    for step,item in enumerate(data):
        # Add the item to the reservoir (stochastically if we're past capacity)
        if step < reservoir_size:
            reservoir.append(item)
            reservoir_ids.append(step)
        else:
            p = np.random.randint(step)
            if p < reservoir_size:
                reservoir[p] = item
                reservoir_ids[p] = step

    return (ma.masked_array(reservoir), reservoir_ids)

def kmeans_plusplus_init(data, weights, k, num_threads, reservoir_size=None, max_steps=None,
                         condor=False, condor_username=None, condor_workers=1500,
                         condor_pollwait=30):
    # Sanity checks
    assert(max_steps is None or max_steps > 1)
    assert(reservoir_size is None or reservoir_size > k)

    if reservoir_size is not None:
        # Save the original data (stream) for later
        original_data = data

        print '\tSampling reservoir of size {0}'.format(reservoir_size)

        # Sample a subset of the data to speed things up.
        data, original_ids = down_sample(data, reservoir_size)

    # Initialize the cluster assignments
    assignments = np.zeros(len(data), dtype=int) - 1

    # Choose the first cluster at random
    available = [i for i in xrange(len(data))]
    chosen = np.random.choice(available)

    # Remove the data point from the list of available data points
    del available[chosen]
    assignments[chosen] = 0

    # Initialize the centroids with the first cluster in it
    centroids = ma.masked_array([data[chosen]])
    
    # Each cluster gets its own ID
    cur_cluster = 1

    # Create K centroids
    while len(centroids) < k and (max_steps is None or cur_cluster < max_steps):
        print '\tCreating centroid #{0}'.format(len(centroids))

        if condor:
            distances = condor_find_nearest_cluster(condor_username, data, weights, centroids, assignments, condor_workers, 0, polling_delay=condor_pollwait, assign=False)
        else:
            distances = parallel_find_nearest_cluster(data, weights, centroids, assignments, num_threads, 0, assign=False)

        # Only use distances from the available centroid candidates
        distances = distances[available]

        # If any values were NaN, convert them to -1
        distances[np.isnan(distances)] = -1.

        # Add 1 to the cosine similarity score to make it in the range [0,2] (so we can square it)
        distances += 1.

        # Take the square of the distance
        distances = distances * distances

        # Normalize the distances to create a probability distribution over points
        distances /= distances.sum()

        # Choose the next centroid via a weighted sample
        chosen = np.random.choice(available, p=distances)

        # Create a new list of centroids with the new one added
        centroids = ma.masked_array(list(centroids) + [data[chosen]])

        # Remove the data point from the list of available data points
        del available[chosen]
        assignments[chosen] = cur_cluster

        # Increment the cluster count
        cur_cluster += 1

    if len(centroids) < k:
        # Figure out how many clusters we still need to create
        remaining = k - len(centroids)

        # Remove the last chosen element from the distances
        if len(available) < len(distances):
            distances = np.delete(distances, chosen)
            distances /= distances.sum()

        print '\tSampling the remaining {0} clusters simultaneously'.format(remaining)

        # Choose the remaining centroids via a weighted sample of the last set of distances
        chosen = np.random.choice(available, p=distances, replace=False, size=remaining)

        # Create a new list of centroids with the new one added
        centroids = ma.masked_array(list(centroids) + [data[c] for c in chosen])

    if reservoir_size is not None:
        print '\tMapping back from the reservoir to the original dataset'

        reservoir_assignments = assignments

        # Create the full-size assignments array
        assignments = np.zeros(len(original_data), dtype=int) - 1

        # Assign all the prototypes to their centroids
        assignments[original_ids] = reservoir_assignments

    # Return the initial assignments and the K centroids
    return (assignments, centroids)

def streaming_centroids(data, assignments, num_clusters):
    '''Calculate the centroids for the data by iterating over the entire dataset'''
    # Create the k x p centroids matrix
    centroids = ma.zeros((num_clusters, data.shape[1]))
    centroids.mask=True

    # Track how many items are in each cluster so we can calculate the mean online
    cluster_sizes = np.zeros(num_clusters)

    # Make a single pass over the dataset to calcluate all centroids
    for step, vec in enumerate(data):
        # Get the cluster to which this vector belongs
        cluster = assignments[step]

        # Update the average vector (centroid) for this cluster
        centroids[cluster] = ma.average([centroids[cluster], vec], axis=0, weights=[cluster_sizes[cluster], 1.])

        # Increment the membership size of the cluster
        cluster_sizes[cluster] += 1.

    # Return the resulting cluster centroids
    return centroids

def streaming_choose_initial_centroids(data, chosen):
    cur = 0
    results = []
    for i,vector in enumerate(data):
        if i == chosen[cur]:
            results.append(vector)
            cur += 1
            if cur == len(chosen):
                break
    return np.ma.masked_array(results, mask=[x.mask for x in results])

def weighted_kmeans(data, weights, k, max_steps, num_threads=4, centroids=None,
                    print_freq=50000, stream=False, pp_init=False,
                    pp_reservoir_size=None, pp_max=None, condor=False,
                    condor_username=None, condor_workers=1500, condor_pollwait=30):
    '''
    Runs a weighted version of k-means that is capable of handling missing
    data. The weights vector is 1xD and data is an NxD matrix. If a centroid is
    missing any data along a dimension, it's set to zero since all dimensions
    are assumed to be standardized to mean-zero.
    '''
    print '\tCreating assignment vector'
    # Track the cluster assignments.
    assignments = np.zeros(len(data), dtype=np.int64) - 1

    # If we were not given centroids, choose k at random.
    if centroids is None:
        if pp_init:
            assignments, centroids = kmeans_plusplus_init(data, weights, k, num_threads,
                                                          reservoir_size=pp_reservoir_size,
                                                          max_steps=pp_max,
                                                          condor=condor,
                                                          condor_username=condor_username,
                                                          condor_workers=condor_workers,
                                                          condor_pollwait=condor_pollwait)
        else:
            print '\tChoosing initial centroids randomly'
            # Choose the k centroids.
            chosen = np.array(sorted(np.random.choice(len(data), k, replace=False)))

            # Track the assignments.
            assignments[chosen] = np.arange(k)
            # Create the centroids.
            if stream:
                centroids = streaming_choose_initial_centroids(data, chosen)
            else:
                centroids = ma.masked_array(data[chosen], mask=data.mask[chosen])
            # Set missing dimensions to zero, since we assume all dimensions
            # are standardized to zero-mean.
            centroids[centroids.mask == True] = 0.
        
    # Track the last iteration's assignments to determine if we've converged.
    prev_assignments = np.copy(assignments)

    # Perform max_steps iterations of k-means.
    for step in xrange(max_steps):
        print '\tIteration #{0}'.format(step+1)

        # Find the assignments for each of the data points
        if condor:
            condor_find_nearest_cluster(condor_username, data, weights, centroids, assignments, condor_workers, step, polling_delay=condor_pollwait)
        else:
            parallel_find_nearest_cluster(data, weights, centroids, assignments, num_threads, step)
        
        # Recalculate all the cluster centroids
        if stream:
            # If we're streaming, it means we want to minimize loops over the data
            centroids = streaming_centroids(data, assignments, k)
        else:
            for i in xrange(k):
                # Take the mean of every dimension.
                centroids[i] = data[assignments==i].mean(axis=0)

        # Set missing dimensions to zero, since we assume all dimensions
        # are standardized to zero-mean.
        centroids[centroids.mask] = 0.

        print 'First few assignments: {0}'.format(assignments[0:20])
        print 'Cluster sizes: {0}'.format(np.bincount(assignments))

        # Check if we've converged to a local optimum.
        if np.array_equal(prev_assignments, assignments):
            break

        # Update the previous list of assignments for comparison with the next
        # iteration.
        prev_assignments = np.copy(assignments)

    return (assignments, centroids)



def save_centroids(centroids, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(centroids.shape)
        writer.writerows(centroids)

def main():
    parser = argparse.ArgumentParser(description='Performs parallel weighted K-means++ vectors.')
    
    # The parameters
    parser.add_argument('vectors', help='The file containing the vectors real-valued numbers.')
    parser.add_argument('cluster_centroids_outfile', help='The file where the cluster centroids will be saved.')
    parser.add_argument('cluster_assignments_outfile', help='The file where the mappings from food ID to cluster ID will be saved.')
    parser.add_argument('--num_clusters', type=int, default=100, help='The number of clusters to create.')
    parser.add_argument('--max_steps', type=int, default=50, help='The maximum number of K-means iterations.')
    parser.add_argument('--num_threads', type=int, default=8, help='The number of threads to use when calculating distances.')
    parser.add_argument('--weights', nargs='+', type=float, help='The weight to give each element of the vector.')
    parser.add_argument('--stream', dest='stream', action='store_true', help='Stream the data rather than loading it all into memory.')
    parser.add_argument('--missing', default='', help='The missing value string for the vectors file.')

    # K-menas++ parameters
    parser.add_argument('--plusplus', '--pp', dest='plusplus', action='store_true', help='Use K-means++ to initialize the clusters.')
    parser.add_argument('--pp_reservoir', type=int, default=1000000, help='The reservoir size for K-means++ initialization.')
    parser.add_argument('--pp_max', type=int, default=100, help='The maximum number of steps to fully recalculate distances when using K-means++ initialization. After pp_max clusters, the remaining clusters will be sampled proportional to the last weight updates.')
    
    # Condor parameters
    parser.add_argument('--condor', dest='condor', action='store_true', help='Use condor to run distributed k-means.')
    parser.add_argument('--condor_username', default='', help='Your username on condor.')
    parser.add_argument('--condor_workers', type=int, default=1500, help='The number of condor jobs to run at once.')
    parser.add_argument('--condor_pollwait', type=int, default=30, help='The number of seconds to wait between polling for condor jobs being finished.')

    parser.set_defaults(stream=False, plusplus=False, condor=False)

    args = parser.parse_args()

    print 'Loading vectors from {0}'.format(args.vectors)
    if args.stream:
        vectors = VectorStream(args.vectors)
        weights = np.array(args.weights) if args.weights else np.ones(vectors._vecsize)
    else:
        x = np.genfromtxt(args.vectors, delimiter=',')
        vectors = np.ma.masked_array(x, mask=np.isnan(x), missing=args.missing)
        weights = np.array(args.weights) if args.weights else np.ones(vectors.shape[1])

    assert vectors.shape[1] == weights.shape[0]

    print 'Clustering vectors'
    assignments, centroids = weighted_kmeans(vectors, weights, args.num_clusters,
                                             args.max_steps, args.num_threads,
                                             stream=args.stream, pp_init=args.plusplus,
                                             pp_reservoir_size=args.pp_reservoir, pp_max=args.pp_max,
                                             condor=args.condor, condor_username=args.condor_username,
                                             condor_workers=args.condor_workers,
                                             condor_pollwait=args.condor_pollwait)
    
    print 'Saving cluster centroids to {0}'.format(args.cluster_centroids_outfile)
    np.savetxt(args.cluster_centroids_outfile, centroids, delimiter=',')

    print 'Saving assignments to {0}'.format(args.cluster_assignments_outfile)
    np.savetxt(args.cluster_assignments_outfile, assignments, delimiter=',', fmt='%d')


if __name__ == '__main__':
    main()


