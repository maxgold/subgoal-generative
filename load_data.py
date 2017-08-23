import h5py
import json
import time
import os



def load_dataset(dataset_name, tmp_dir="/tmp"):
    print ("Loading dataset...")
    dataset_filename = "{}.hdf5".format(dataset_name)
    dataset_filepath = "{}/data/dataset/{}".format('/Users/maxgold/rll/planning_networks', dataset_filename)
    filepath_gz = "{}.tar.gz".format(dataset_filepath)
    if os.path.exists(filepath_gz):
        dataset_filepath = "{}/{}".format(tmp_dir, dataset_filename)
        if not os.path.exists(dataset_filepath):
            start = time.time()
            with tarfile.open(filepath_gz, mode='r') as f:
                f.extractall(tmp_dir)
            print ("Dataset decompression took: {} seconds".format(time.time()-start))
    return h5py.File(dataset_filepath, 'r')







