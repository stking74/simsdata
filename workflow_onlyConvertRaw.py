# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:26:08 2016

@author: iv1
"""
import datetime

#%%This code won’t work from Python console
#!!! Clean import procedure with __init__.py file etc.
from SIMSmodel import SIMSmodel
from SIMSdata import SIMSdata
verbose = True
#%%

#print('start time = %s'%(str(timestamps['start'])))
if __name__ == '__main__':

    cores = 8
    #Path to file and prefix
    path = r'F:\kick-ass brain data\SIMS'
    prefix = '001 stage scan mouse brain'
    #Initialize converter class
    if verbose:
        print('Initializing data handler...')
    sims_data = SIMSdata()
    #sims_data.load_h5(h5_path)
    
    #Load SIMS measurement data from raw datafile
    if verbose:
        print('Loading raw data...')
    sims_data.load_raw(path, prefix)

    try:
        sims_data.h5f.flush()
        sims_data.h5f.close()
    except: pass

