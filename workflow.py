# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:26:08 2016

@author: iv1
"""
import datetime
import os
root = os.getcwd()
libdir = r'/home/tyler/Code/Python'
os.chdir(libdir)
import simsdata
os.chdir(root)

verbose = True

#%%

#print('start time = %s'%(str(timestamps['start'])))
if __name__ == '__main__':

    cores = 1
    #Path to file and prefix
    path = r'/home/tyler/Documents/Work/Ovchinnikova/hackathon test data'
    prefix = '003 posions dark 0v'
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
    model = simsdata.SIMSmodel(xy_bins=4, z_bins=1, counts_threshold=0.001, tof_resolution=64, cores=cores) #Minimal overhead

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
