# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:26:08 2016

@author: iv1
"""
import os
root = os.getcwd()
libdir = r'C:\Users\s4k\Documents\Python'
os.chdir(libdir)
import simsdata
os.chdir(root)

verbose = True

#%%

#print('start time = %s'%(str(timestamps['start'])))
if __name__ == '__main__':

    cores = 8
    #Path to file and prefix
    path = r'E:\TylerData\temp'
    prefix = '004 stage scan mouse brain posions'
    #Initialize converter class
    if verbose:
        print('Initializing data handler...')
    sims_data = simsdata.SIMSdata()
    #sims_data.load_h5(h5_path)
    
    #Load SIMS measurement data from raw datafile
    if verbose:
        print('Loading raw data...')
    sims_data.load_raw(path, prefix)

    #Preparing SIMS conversion model
    if verbose:
        print('Intiailizing data converter...')
    # xy_bins: width of bins to use for x/y binning
    # z_bins: width of bins to use for z binning
    # counts_threshold: sets value for intensity thresholding of raw peak data
        # If > 1, interpreted as absolute threshold. Peaks with intensity < counts_threshold will not pass filter.
        # Elif >= 0 and < 1, interpreted as relative threshold. Peaks with intensity < (counts_threshold * base_peak_intensity) will not pass fiter.
    # tof_resolution: width of bins to use for tof binning
    # cores: number of CPU cores to use for parallel processing
    # chunk_size: HDF5 chunk length, also size of single processing work unit
    model = simsdata.SIMSmodel(tof_resolution=64, xy_bins=25, xy_resolution=2, z_bins=1, z_resolution=1, counts_threshold=0.01, cores=cores, chunk_size=1e6) 

    model.convert_all_peaks()
    #model.enable_shift_correction()
    if verbose:
        print('Converting data...')
    data2d = sims_data.convert_data('all', model, cores)
#    if verbose:
#        print('Loading converted data...')
#    sims_data.load_converted_data(1)
#    if verbose:
#        print('Calculating PCA...')
#    sims_data.PCA(comp_num=20, spatial_range=(slice(30), -1, -1))
#    if verbose:
#        print('Calculating NMF...')
#    sims_data.NMF(20)

    #Plotting averaged data
    #sims_data.plot_ave_data()
    if verbose:
        print('Finished converting, closing model.')
    sims_data.close()
