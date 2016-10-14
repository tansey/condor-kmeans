A Condor-powered K-means implementation
---------------------------------------
<p align="center">
  <img src="https://github.com/tansey/condor-kmeans/blob/master/test/results.png?raw=true" alt="Example K-means Solution"/>
</p>


This package lets you run K-means on a really big dataset of vectors. You can even stream the vectors instead of loading them into memory, so long as you can store two lists of doubles the size of your vector count (one list for cluster assignment IDs and one for distance from each vector to its cluster).

## Installation

Installation is available via `pip`:

```
pip install condor-kmeans
```

## Usage

The package assumes you have a CSV file of vectors which you wish to cluster, with one vector per row. Once installed, you can simply run the `kmeans` command:

```
kmeans path/to/mydata.csv path/to/save/centroids.csv path/to/save/assignments.csv --num_clusters 30 --plusplus --stream --condor --condor_workers 100 --condor_username myusername
```

The above command will run k-means on the vectors stored in `mydata.csv` on condor with no more than 100 jobs at a time. It will save the resulting cluster centroids to `centroids.csv`, and the resulting vector-to-cluster assignments to `assignments.csv`. The `--plusplus` command specifies it should use k++ initialization. `--stream` says to stream `mydata.csv` from disk instead of loading it all into memory.

The current directory is used as the working directory. A working subdirectory named `condor` will be created. All temporary worker files are deleted after each batch of jobs is finished successfully, though the directory structure is maintained (feel free to just `rm -rf condor` afterward if you wish). If one of the workers fails, the master will throw an exception and alert you to the job that failed and where to find its output files; the temporary files will not be deleted if a worker fails.