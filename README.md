# SIMSdata

A Python reformatting and processing pipeline for ToF-SIMS imaging mass spectrometry data. 

### About

This package was developed by the Center for Nanophase Materials Sciences at Oak Ridge National Laboratory to facillitate restructuring of data acquired ION-TOF ToF-SIMS instrumentation. The workflow is designed to accept binary data files exported from IONTOF software, process the data files to identify mass spectral peaks, and output a 4-dimensional data hypercube as a HDF5 file for use in further analysis. 

## Getting Started

### Dependencies

This package was built with the Anaconda distribution of Python 3.6. All required packages and modules are included with [Anaconda](https://anaconda.org/).

The full list of required packages is:
* Numpy
* Scipy
* scikit-learn
* h5py
* matplotlib

### Installation

Download the module and save to a directory included in your machine's PYTHONPATH variable, or otherwise [add the directory to the machine's PYTHONPATH](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath). Alternatively, the path to the module can be specified at runtime using line 10 of workflow.py.

### Usage

* Open the file workflow.py in a text editor
* Enter the directory which contains the SIMSdata module as "libdir" in line 10
* Enter the number of CPU cores to use for processing as "cores" in line 22
* Enter the directory which contains the data files to be processed as "path" in line 24
* Enter the filename prefix (PREFIX ONLY) of the data files to be processed as "prefix" in line 25
* In line 40, specify parameters for xy_bins, z_bins, counts_threshold, tof_resolution, and chunk_size for the SIMS data model.
* Save and execute workflow.py. The output HDF5 file will be created in the directory entered as "path"
