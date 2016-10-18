import sys
import os
import numpy as np
import numpy.ma as ma
import argparse
import csv
import time
import traceback
from condor_kmeans.vector import VectorStream
from condor_kmeans.utils import make_directory

JOB_HEADER = '''universe = vanilla
Executable=/lusr/bin/python
Requirements = InMastodon
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Distributed K-means"
getenv = True

'''

FIND_CLUSTER_MAP_JOB = '''Arguments = {python_filepath} worker $(workerid) $(step) {data_filename} $(start) $(end) {weights_filename} {data_dir}step$(step)_centroids.csv {assignments_dir}$(workerid).csv {mindistance_dir}$(workerid).csv {partial_centroids_dir}$(workerid).csv {partial_centroids_counts_dir}$(workerid).csv {finished_flag}
Output = {output_dir}find_nearest_cluster_$(workerid)_step$(step).out
Error = {error_dir}find_nearest_cluster_$(workerid)_step$(step).err
Queue 1

'''

AGGREGATE_JOB = '''Arguments = {python_filepath} aggregate $(step) {data_filename} {username} {num_workers} {working_dir} {max_steps} {finished_flag} {final_centroids_outfile} {final_assignments_outfile}
Output = {output_dir}aggregate_step$(step).out
Error = {error_dir}aggregate_step$(step).err
Queue 1

'''


class CondorKmeans(object):
    def __init__(self, username, num_workers, working_dir, final_centroids_outfile, final_assignments_outfile, max_steps=50):
        self._username = username
        self._num_workers = num_workers
        self._working_dir = working_dir
        self._base_dir = make_directory(working_dir, 'condor')
        self._data_dir = make_directory(self._base_dir, 'data')
        self._final_centroids_outfile = final_centroids_outfile
        self._final_assignments_outfile = final_assignments_outfile
        self._max_steps = max_steps

    def _get_worker_ranges(self, data):
        # Figure out how many data points each worker should be working on
        num_workers = min(len(data), self._num_workers)
        worker_load = len(data) / num_workers

        # Each worker is given a range in the form (inclusive_start, exclusive_end)
        worker_ranges = [(i*worker_load, (i+1)*worker_load) for i in xrange(num_workers)]
        worker_ranges[-1] = (worker_ranges[-1][0], len(data))

        return worker_ranges

    def _get_dargs(self, step, data, worker_id=0, start=0, end=0):
        # Create a map for use in string formatting for the jobs file
        dargs = {'username': self._username, 'base_dir': self._base_dir, 'python_filepath': os.path.abspath(__file__) }
        dargs['working_dir'] = self._working_dir
        dargs['data_dir'] = self._data_dir
        dargs['num_workers'] = self._num_workers
        dargs['final_centroids_outfile'] = self._final_centroids_outfile
        dargs['final_assignments_outfile'] = self._final_assignments_outfile
        dargs['max_steps'] = self._max_steps
        dargs['step'] = step
        dargs['worker_id'] = worker_id
        dargs['start'] = start
        dargs['end'] = end
        # Get the location of the data
        if data is VectorStream:
            dargs['data_filename'] = data._filename
        else:
            # If we were given data in memory instead of in a stream, save it to file so it can be streamed
            dargs['data_filename'] = self._data_dir + 'data.csv'
        # Setup the file structure
        dargs['output_dir'] = make_directory(self._base_dir, 'output')
        dargs['error_dir'] = make_directory(self._base_dir, 'error')
        dargs['job_dir'] = make_directory(self._base_dir, 'jobs')
        dargs['assignments_dir'] = make_directory(self._data_dir, 'assignments')
        dargs['mindistance_dir'] = make_directory(self._data_dir, 'mindistance')
        dargs['partial_centroids_dir'] = make_directory(self._data_dir, 'partial_centroids')
        dargs['partial_centroids_counts_dir'] = make_directory(self._data_dir, 'partial_centroids_counts')
        dargs['weights_filename'] = self._data_dir + 'weights.csv'
        dargs['centroids_filename'] = self._data_dir + 'step{step}_centroids.csv'.format(**dargs)
        dargs['aggregated_assignments_filename'] = self._data_dir + 'step{step}_assignments.csv'.format(**dargs)
        dargs['assignments_outfile'] = '{assignments_dir}{worker_id}.csv'.format(**dargs)
        dargs['mindistance_outfile'] = '{mindistance_dir}{worker_id}.csv'.format(**dargs)
        dargs['partial_centroids_outfile'] = '{partial_centroids_dir}{worker_id}.csv'.format(**dargs)
        dargs['partial_centroids_counts_outfile'] = '{partial_centroids_counts_dir}{worker_id}.csv'.format(**dargs)
        dargs['output_filename'] = '{output_dir}find_nearest_cluster_{worker_id}_step{step}.out'.format(**dargs)
        dargs['error_filename'] = '{error_dir}find_nearest_cluster_{worker_id}_step{step}.err'.format(**dargs)
        dargs['jobs_filename'] = '{job_dir}find_nearest_cluster_jobs'.format(**dargs)
        dargs['agg_output_filename'] = '{output_dir}aggregate_step{step}.out'.format(**dargs)
        dargs['agg_error_filename'] = '{error_dir}aggregate_step{step}.err'.format(**dargs)
        dargs['aggjob_filename'] = '{job_dir}aggregate'.format(**dargs)
        dargs['dagman_filename'] = '{job_dir}dag'.format(**dargs)
        dargs['subdag_filename'] = '{job_dir}subdag'.format(**dargs)
        dargs['finished_flag'] = '{output_dir}finished'.format(**dargs)
        return dargs

    def weighted_kmeans(self, data, weights, k, num_threads=4, centroids=None,
                        pp_init=False, pp_reservoir_size=None, pp_max=None):
        dargs = self._get_dargs(0, data)
        # Get the location of the data
        if data is not VectorStream:
            # If we were given data in memory instead of in a stream, save it to file so it can be streamed
            np.savetxt(dargs['data_filename'], data, delimiter=',')

        # Write the weights to file
        np.savetxt(dargs['weights_filename'], weights, delimiter=',')

        # Write the worker job file
        with open(dargs['jobs_filename'], 'wb') as f:
            f.write(JOB_HEADER.format(**dargs))
            f.write(FIND_CLUSTER_MAP_JOB.format(**dargs))

        # Write the master aggregation job file
        with open(dargs['aggjob_filename'], 'wb') as f:
            f.write(JOB_HEADER.format(**dargs))
            f.write(AGGREGATE_JOB.format(**dargs))

        # Get the start and end regions for each worker job
        worker_ranges = self._get_worker_ranges(data)

        # Generate all the job scripts
        dag_parents = ''
        dag_jobvars = ''
        subdag_parents = ''
        subdag_jobvars = ''
        with open(dargs['dagman_filename'], 'wb') as dagf:
            with open(dargs['subdag_filename'], 'wb') as subdagf:
                # Create the initial centroids if none are given
                if centroids is None:
                    # Use Kmeans++ initialization
                    if pp_init:
                        for step in xrange(k if pp_max is None else min(k, pp_max)):
                            with open(dargs['initjob_filename']):
                                pass # TODO
                    else:
                        # If not using Kmeans++, just randomly pick centroids (this seems to often work better)
                        from condor_kmeans.kmeans import choose_random_centroids
                        centroids = choose_random_centroids(data, k, stream=data is VectorStream)
                        np.savetxt(dargs['centroids_filename'], centroids, delimiter=',')
                else:
                    # If we're given some centroids, use those instead
                    np.savetxt(dargs['centroids_filename'], centroids, delimiter=',')

                # Write the top-level dag for the steps
                for step in xrange(self._max_steps):
                    dargs = self._get_dargs(step, data)
                    dargs['job_id'] = 'STEP{step}'.format(**dargs)
                    dagf.write('SUBDAG EXTERNAL {job_id} {subdag_filename}\n'.format(**dargs))
                    dag_jobvars += 'VARS {job_id} step="{step}"\n'.format(**dargs)
                    if step < (self._max_steps-1):
                        dargs['next_step'] = step + 1
                        dag_parents += 'PARENT STEP{next_step} {job_id}\n'.format(**dargs)
                    

                # Write all the dependencies
                dagf.write(dag_jobvars)
                dagf.write(dag_parents)

                # Write the sub-dag that handles each worker in a given step
                for i, (start, end) in enumerate(worker_ranges):
                    dargs = self._get_dargs(0, data, i, start, end)
                    # Open up a jobs file
                    dargs['subdag_job_id'] = 'FINDCLUSTERS{worker_id}'.format(**dargs)
                    subdagf.write('JOB {subdag_job_id} {jobs_filename}')
                    subdag_jobvars += 'VARS {subdag_job_id} step="$(step)"\n'.format(**dargs)
                    subdag_jobvars += 'VARS {subdag_job_id} workerid="{worker_id}"\n'.format(**dargs)
                    subdag_jobvars += 'VARS {subdag_job_id} start="{start}"\n'.format(**dargs)
                    subdag_jobvars += 'VARS {subdag_job_id} end="{end}"\n'.format(**dargs)
                    dag_parents += 'PARENT {subdag_job_id} CHILD AGG\n'.format(**dargs)
                subdagf.write('JOB AGG {aggjob_filename}\n'.format(**dargs))
                subdag_jobvars += 'VARS AGG step="$(step)"\n'.format(**dargs)

                # Write all the dependencies
                subdagf.write(subdag_jobvars)
                subdagf.write(subdag_parents)
                

        # Check if there is a leftover finished flag and remove it
        if os.path.exists(dargs['finished_flag']):
            os.remove(dargs['finished_flag'])

        # Submit the jobs to condor. Fuck using subprocess at the moment
        os.system('condor_submit_dag {dagman_filename}'.format(**dargs))

    def check_workers_finished(self, step, data):
        worker_ranges = self._get_worker_ranges(data)

        # Check to see if any workers failed file exists
        for i, (start, end) in enumerate(worker_ranges):
            dargs = self._get_dargs(step, data, i, start, end)

            # If the worker hasn't even started yet, chill some more
            if not os.path.exists(dargs['output_filename']):
                raise Exception('Worker output file {output_filename} does not exist.'.format(**dargs))
                break

            # Check the output of the file
            with open(dargs['output_filename'], 'rb') as f:
                lines = f.readlines()

                # If the file hasn't been written to yet, chill some more
                if len(lines) == 0:
                    raise Exception('Worker output file {output_filename} is empty.'.format(**dargs))
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

        print 'Workers all finished successfully.'

    def aggregate_worker_results(self, step, data, assign=True):
        print 'Aggregating results.'
        #centroids, assignments, min_distances
        min_distances = np.zeros(data.shape[0])

        centroids = None
        assignments = None

        # All the workers are finished. Merge the results back
        worker_ranges = self._get_worker_ranges(data)
        for i, (start, end) in enumerate(worker_ranges):
            dargs = self._get_dargs(step, data, i, start, end)

            # Check the output of the file
            min_distances[start:end] = np.loadtxt(dargs['mindistance_outfile'], delimiter=',')
            if assign:
                if assignments is None:
                    assignments = np.zeros(data.shape[0], dtype=int)
                # Get the assignments for the subset assigned to this worker
                assignments[start:end] = np.loadtxt(dargs['assignments_outfile'], delimiter=',')

                # Get the partial centroids calculated from the subset of the data for this worker
                partial_centroids = np.loadtxt(dargs['partial_centroids_outfile'], delimiter=',')
                partial_centroids_counts = np.loadtxt(dargs['partial_centroids_counts_outfile'], delimiter=',')

                # Create the centroids array
                if centroids is None:
                    centroids = np.ma.masked_array(np.zeros(partial_centroids.shape), mask=np.zeros(partial_centroids.shape, dtype=int))
                    centroids_counts = np.zeros(centroids.shape[0])

                # Calculate a running mean for the cluster centers
                next_counts = centroids_counts + partial_centroids_counts
                centroids = (centroids * (centroids_counts / next_counts.clip(1))[:,np.newaxis]
                                     + partial_centroids * (partial_centroids_counts / next_counts.clip(1))[:,np.newaxis])
                centroids_counts = next_counts

            # Clean up
            os.remove(dargs['assignments_outfile'])
            os.remove(dargs['mindistance_outfile'])
            os.remove(dargs['partial_centroids_outfile'])
            os.remove(dargs['partial_centroids_counts_outfile'])
            os.remove(dargs['output_filename'])
            os.remove(dargs['error_filename'])

        if assign:
            np.savetxt(dargs['aggregated_assignments_filename'], assignments, delimiter=',', fmt='%d')
            dargs = self._get_dargs(step+1, data)
            np.savetxt(dargs['centroids_filename'], centroids, delimiter=',')

            if step > 0:
                dargs = self._get_dargs(step-1, data)
                prev_assignments = np.loadtxt(dargs['aggregated_assignments_filename'], delimiter=',')
                # Check if we've converged to a local optimum.
                if step == (self._max_steps-1) or np.array_equal(prev_assignments, assignments):
                    with open(dargs['finished_flag'], 'wb') as f:
                        f.write('Finished!')

                    # Save the clustering results to their final destinations
                    np.savetxt(dargs['final_centroids_outfile'], centroids, delimiter=',')
                    np.savetxt(dargs['final_assignments_outfile'], assignments, delimiter=',', fmt='%d')
                    
                    # If we had to write the data to file, delete the temp file
                    if data is not VectorStream:
                        os.remove(dargs['data_filename'])

            return min_distances, assignments

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
    finished_flag = sys.argv[13]

    if os.path.exists(finished_flag):
        exit(0)

    try:
        data = VectorStream(data_filename)
        weights = np.loadtxt(weights_filename, delimiter=',')
        centroids = np.loadtxt(centroids_filename, delimiter=',')
        assignments = np.zeros(end-start, dtype=int) - 1

        if len(centroids.shape) == 1:
            centroids = centroids[:, np.newaxis].T
    except Exception as ex:
        print traceback.format_exc()
        print 'Error loading files: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        # Reuse the local kmeans nearest-cluster routine
        from condor_kmeans.kmeans import find_nearest_cluster
        all_params = (step, data, start, end, weights, centroids, assignments, worker_id)
        (worker_id, assignments, min_distances) = find_nearest_cluster(all_params)
    except Exception as ex:
        print traceback.format_exc()
        print 'Error finding nearest cluster: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        # Calculate the cluster centroids and counts
        print 'Calculating partial centroids'
        partial_centroids = np.ma.masked_array(np.zeros(centroids.shape), mask=np.zeros(centroids.shape, dtype=int))
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
        print traceback.format_exc()
        print 'Error calculating partial centroids: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        print 'Writing assignments to {0} (shape: {1})'.format(assignments, assignments.shape)
        print 'Writing min distances to {0} (shape: {1})'.format(mindistance_outfile, min_distances.shape)
        np.savetxt(assignments_outfile, assignments, delimiter=',')
        np.savetxt(mindistance_outfile, min_distances, delimiter=',')
        np.savetxt(partial_centroids_outfile, partial_centroids, delimiter=',')
        np.savetxt(partial_centroids_counts_outfile, centroid_counts, delimiter=',')
    except Exception as ex:
        print traceback.format_exc()
        print 'Error saving files: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    print 'Success!'

def aggregate_main():
    step = int(sys.argv[2])
    data_filename = sys.argv[3]
    username = sys.argv[4]
    num_workers = int(sys.argv[5])
    working_dir = sys.argv[6]
    max_steps = int(sys.argv[7])
    finished_flag = sys.argv[8]
    final_centroids_outfile = sys.argv[9]
    final_assignments_outfile = sys.argv[10]

    if os.path.exists(finished_flag):
        exit(0)

    try:
        data = VectorStream(data_filename)
    except Exception as ex:
        print traceback.format_exc()
        print 'Error loading files: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)

    try:
        pool = CondorKmeans(username, num_workers, working_dir, final_centroids_outfile, final_assignments_outfile, max_steps=max_steps)

        pool.check_workers_finished(step, data)

        pool.aggregate_worker_results(step, data, assign=True)
    except Exception as ex:
        print traceback.format_exc()
        print 'Error aggregating worker results: {0}'.format(ex)
        print 'Quit due to error'
        exit(1)


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        worker_main()
    elif sys.argv[1] == 'aggregate':
        aggregate_main()
    























