import sys
import os
import numpy as np
import numpy.ma as ma
import argparse
import csv
import time
from condor_kmeans.vector import VectorStream
from condor_kmeans.utils import make_directory

FIND_CLUSTER_MAP_HEADER = '''universe = vanilla
Executable=/lusr/bin/python
Requirements = InMastodon
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Distributed K-means"
getenv = True

'''

FIND_CLUSTER_MAP_JOB = '''Arguments = {python_filepath} worker {worker_id} {step} {data_filename} {start} {end} {weights_filename} {centroids_filename} {assignments_outfile} {mindistance_outfile} {partial_centroids_outfile} {partial_centroids_counts_outfile}
Output = {output_filename}
Error = {error_filename}
Queue 1

'''


class CondorKmeansPool(object):
    def __init__(self, username, num_workers, working_dir, polling_delay_in_seconds=30, timeout_in_seconds=60*60*24):
        self._username = username
        self._num_workers = num_workers
        self._base_dir = make_directory(working_dir, 'condor')
        self._data_dir = make_directory(self._base_dir, 'data')
        self._polling_delay_in_seconds = polling_delay_in_seconds

    def map_find_nearest_cluster(self, step, data, weights, centroids, assignments, min_distances, assign=True):
        # Create a map for use in pretty string formatting for the jobs file
        dargs = {'username': self._username, 'step': step, 'base_dir': self._base_dir, 'python_filepath': os.path.abspath(__file__) }

        # Get the location of the data
        if data is VectorStream:
            dargs['data_filename'] = data._filename
        else:
            # If we were given data in memory instead of in a stream, save it to file so it can be streamed
            dargs['data_filename'] = self._data_dir + 'data.csv'
            np.savetxt(dargs['data_filename'], data, delimiter=',')

        # Setup the file structure
        dargs['output_dir'] = make_directory(self._base_dir, 'output')
        dargs['error_dir'] = make_directory(self._base_dir, 'error')
        dargs['assignments_dir'] = make_directory(self._data_dir, 'assignments')
        dargs['mindistance_dir'] = make_directory(self._data_dir, 'mindistance')
        dargs['partial_centroids_dir'] = make_directory(self._data_dir, 'partial_centroids')
        dargs['partial_centroids_counts_dir'] = make_directory(self._data_dir, 'partial_centroids_counts')

        # Write the weights to file
        dargs['weights_filename'] = self._data_dir + 'weights.csv'
        np.savetxt(dargs['weights_filename'], weights, delimiter=',')
        
        # Write the centroids to file
        dargs['centroids_filename'] = self._data_dir + 'step{step}_centroids.csv'.format(**dargs)
        np.savetxt(dargs['centroids_filename'], centroids, delimiter=',')

        # Figure out how many data points each worker should be working on
        num_workers = min(len(data), self._num_workers)
        worker_load = len(data) / num_workers

        # Each worker is given a range in the form (inclusive_start, exclusive_end)
        worker_ranges = [(i*worker_load, (i+1)*worker_load) for i in xrange(num_workers)]
        worker_ranges[-1] = (worker_ranges[-1][0], len(data))

        # Open up a jobs file
        dargs['jobs_filename'] = '{base_dir}find_nearest_cluster_jobs'.format(**dargs)
        with open(dargs['jobs_filename'], 'wb') as f:
            # Write the header to the top of the file
            f.write(FIND_CLUSTER_MAP_HEADER.format(**dargs))

            # Write each worker's job to file
            for i, (start, end) in enumerate(worker_ranges):
                dargs['worker_id'] = i
                dargs['start'] = start
                dargs['end'] = end
                dargs['assignments_outfile'] = '{assignments_dir}{worker_id}.csv'.format(**dargs)
                dargs['mindistance_outfile'] = '{mindistance_dir}{worker_id}.csv'.format(**dargs)
                dargs['partial_centroids_outfile'] = '{partial_centroids_dir}{worker_id}.csv'.format(**dargs)
                dargs['partial_centroids_counts_outfile'] = '{partial_centroids_counts_dir}{worker_id}.csv'.format(**dargs)
                dargs['output_filename'] = '{output_dir}find_nearest_cluster_{worker_id}.out'.format(**dargs)
                dargs['error_filename'] = '{error_dir}find_nearest_cluster_{worker_id}.err'.format(**dargs)
                f.write(FIND_CLUSTER_MAP_JOB.format(**dargs))

        # Submit the job to condor. Fuck using subprocess at the moment
        os.system('condor_submit {jobs_filename}'.format(**dargs))

        print 'Jobs submitted. Polling for completion...'

        # Flag to keep waiting
        poll = True

        # Poll and chill
        # TODO: implement timeout counter
        while poll:
            # Chill
            time.sleep(self._polling_delay_in_seconds)

            poll = False

            # Check to see if each file exists
            for i, (start, end) in enumerate(worker_ranges):
                dargs['worker_id'] = i
                dargs['start'] = start
                dargs['end'] = end
                dargs['assignments_outfile'] = '{assignments_dir}{worker_id}.csv'.format(**dargs)
                dargs['mindistance_outfile'] = '{mindistance_dir}{worker_id}.csv'.format(**dargs)
                dargs['partial_centroids_outfile'] = '{partial_centroids_dir}{worker_id}.csv'.format(**dargs)
                dargs['partial_centroids_counts_outfile'] = '{partial_centroids_counts_dir}{worker_id}.csv'.format(**dargs)
                dargs['output_filename'] = '{output_dir}find_nearest_cluster_{worker_id}.out'.format(**dargs)
                dargs['error_filename'] = '{error_dir}find_nearest_cluster_{worker_id}.err'.format(**dargs)

                # If the worker hasn't even started yet, chill some more
                if not os.path.exists(dargs['output_filename']):
                    poll = True
                    break

                # Check the output of the file
                with open(dargs['output_filename'], 'rb') as f:
                    lines = f.readlines()

                    # If the file hasn't been written to yet, chill some more
                    if len(lines) == 0:
                        poll = True
                        break

                    # Get the last non-empty line of the output file
                    text = lines[-1].strip('\n')
                    while text == '' and len(lines) > 0:
                        lines = lines[:-1]
                        text = lines[-1].strip('\n')

                    # If the worker failed, raise an exception in the master and point to the debug files
                    if text == 'Quit due to error':
                        os.system('condor_rm {username}'.format(**dargs))
                        raise Exception('Worker failure occurred. Worker #{worker_id} range: [{start}-{end}] output file: {output_filename} error file: {error_filename}'.format(**dargs))
                    elif text == 'Success!':
                        # If the worker succeeded, move on to the next file
                        continue
                    else:
                        # If the worker hasn't finished writing its output file, chill some more
                        poll = True
                        break

        print 'Workers finished. Aggregating results.'

        if assign:
            updated_centroids = np.ma.zeros(centroids.shape, mask=np.zeros(centroids.shape, dtype=int))
            updated_centroids_counts = np.zeros(centroids.shape[0])

        # All the workers are finished. Merge the results back
        for i, (start, end) in enumerate(worker_ranges):
            dargs['worker_id'] = i
            dargs['start'] = start
            dargs['end'] = end
            dargs['assignments_outfile'] = '{assignments_dir}{worker_id}.csv'.format(**dargs)
            dargs['mindistance_outfile'] = '{mindistance_dir}{worker_id}.csv'.format(**dargs)
            dargs['partial_centroids_outfile'] = '{partial_centroids_dir}{worker_id}.csv'.format(**dargs)
            dargs['partial_centroids_counts_outfile'] = '{partial_centroids_counts_dir}{worker_id}.csv'.format(**dargs)
            dargs['output_filename'] = '{output_dir}find_nearest_cluster_{worker_id}.out'.format(**dargs)
            dargs['error_filename'] = '{error_dir}find_nearest_cluster_{worker_id}.err'.format(**dargs)

            # Check the output of the file
            min_distances[start:end] = np.loadtxt(dargs['mindistance_outfile'], delimiter=',')
            if assign:
                assignments[start:end] = np.loadtxt(dargs['assignments_outfile'], delimiter=',')
                partial_centroids = np.loadtxt(dargs['partial_centroids_outfile'], delimiter=',')
                partial_centroids_counts = np.loadtxt(dargs['partial_centroids_counts_outfile'], delimiter=',')
                # Calculate a running mean for the cluster centers
                next_counts = updated_centroids_counts + partial_centroids_counts
                updated_centroids = (updated_centroids * (updated_centroids_counts / next_counts.clip(1))[:,np.newaxis]
                                     + partial_centroids * (partial_centroids_counts / next_counts.clip(1))[:,np.newaxis])
                updated_centroids_counts = next_counts

            # Clean up
            os.remove(dargs['assignments_outfile'])
            os.remove(dargs['mindistance_outfile'])
            os.remove(dargs['partial_centroids_outfile'])
            os.remove(dargs['partial_centroids_counts_outfile'])
            os.remove(dargs['output_filename'])
            os.remove(dargs['error_filename'])

        os.remove(dargs['jobs_filename'])

        # If we had to write the data to file, delete the temp file
        if data is not VectorStream:
            os.remove(dargs['data_filename'])

        if assign:
            return updated_centroids


def condor_find_nearest_cluster(condor_username, data, weights, centroids, assignments, num_workers, step, assign=True, polling_delay=30):
    min_distances = np.zeros(assignments.shape)

    # Create the thread pool to process things in parallel on condor
    pool = CondorKmeansPool(condor_username, num_workers, os.getcwd(), polling_delay_in_seconds=polling_delay)
    
    # Map the jobs to condor workers
    pool.map_find_nearest_cluster(step, data, weights, centroids, assignments, min_distances, assign=assign)

    return min_distances

def worker_main():
    worker_id = int(sys.argv[2])
    step = int(sys.argv[3])
    data_filename = sys.argv[4]
    start = int(sys.argv[5])
    end = int(sys.argv[6])
    weights_filename = sys.argv[7]
    centroids_filename = sys.argv[8]
    assignments_outfile = sys.argv[9]
    mindistance_outfile = sys.argv[10]
    partial_centroids_outfile = sys.argv[11]
    partial_centroids_counts_outfile = sys.argv[12]

    try:
        data = VectorStream(data_filename)
        weights = np.loadtxt(weights_filename, delimiter=',')
        centroids = np.loadtxt(centroids_filename, delimiter=',')
        assignments = np.zeros(end-start, dtype=int) - 1

        if len(centroids.shape) == 1:
            centroids = centroids[:, np.newaxis].T
    except Exception as ex:
        print 'Error loading files: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        # Reuse the local kmeans nearest-cluster routine
        from diet2vec.clustering.kmeans import find_nearest_cluster
        all_params = (step, data, start, end, weights, centroids, assignments, worker_id)
        (worker_id, assignments, min_distances) = find_nearest_cluster(all_params)
    except Exception as ex:
        print 'Error finding nearest cluster: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        # Calculate the cluster centroids and counts
        print 'Calculating partial centroids'
        partial_centroids = np.zeros(centroids.shape)
        centroid_counts = np.zeros(centroids.shape[0])
        assign_idx = 0
        for i, x in enumerate(data):
            if i < start:
                continue
            if i >= end:
                break
            centroid_idx = assignments[assign_idx]
            partial_centroids[centroid_idx] += x
            centroid_counts[centroid_idx] += 1
            assign_idx += 1
        partial_centroids = partial_centroids / centroid_counts[:,np.newaxis]
    except Exception as ex:
        print 'Error calculating partial centroids: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        np.savetxt(assignments_outfile, assignments, delimiter=',')
        np.savetxt(mindistance_outfile, min_distances, delimiter=',')
        np.savetxt(partial_centroids_outfile, partial_centroids, delimiter=',')
        np.savetxt(partial_centroids_counts_outfile, centroid_counts, delimiter=',')
    except Exception as ex:
        print 'Error saving files: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    print 'Success!'

if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        worker_main()
    elif sys.argv[1] == 'aggregate':
        aggregate_main()
    























