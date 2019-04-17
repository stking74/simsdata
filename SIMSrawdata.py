# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:39:26 2016

@author: iv1
"""
import time

import numpy as np
import h5py
import os

class SIMSrawdata(object):
    '''
    Class stroing and handling raw ToF SIMS data
    '''

    def __init__(self):
        return

    def load_from_file(self, path, file_prefix, nuke=False):
        '''
        Loads raw SIMS data from GRD files

        Inputs
        --------
        path : String or unicode
            Path to folder with GRD files
        file_prefix : String or unicode
            Prefix of the dataset name
        '''
        scans_path=os.path.join(path,file_prefix+'.itm.scans')
        coords_path=os.path.join(path,file_prefix+'.itm.coords')
        tofs_path=os.path.join(path,file_prefix+'.itm.tofs')
        
        h5_path = os.path.join(path, file_prefix+'.h5')
        h5f = h5py.File(h5_path, 'w')
        
        chunk_order = 2                                                         #Base chunk sizing unit, 1 = KB, 2 = MB, 3 = GB
        unit_size = 100                                                           #Multiples of unit per chunk
        chunk_size = int((((8*(10**(3*chunk_order)))/32) * unit_size) / 2)
        
        #Import ToF data from datafile
        f=open(tofs_path, 'rb')
        tofs=np.fromstring(f.read(),dtype='uint32')                        # And again for the .tofs file (20%/80%)
        f.close()
        counts = len(tofs)
        n_blocks = int(counts//chunk_size)
        remainder = int(counts%chunk_size)
        print('Sorting raw data...')
        sii = np.argsort(tofs)
        
        if 'Raw_data' in h5f.keys():
            del h5f['Raw_data']
        grp=h5f.create_group('Raw_data')
        dset=grp.create_dataset('Raw_data', (counts, 4), dtype='uint32', chunks=(chunk_size, 1))
        print('Writing tofs...')
        for i in range(n_blocks):
            low = i * chunk_size
            high = low + chunk_size
            block = [tofs[sii[jj]] for jj in range(low, high)]
            dset[low:high, 3] = block
            del block
        low = n_blocks * chunk_size
        high = low + remainder
        block = [tofs[sii[jj]] for jj in range(low, high)]
        dset[low:high, 3] = block
        del tofs, block
        h5f.flush()
        if nuke: os.remove(tofs_path)
        sii = np.array(sii, dtype='uint32')
        #Read measurement coordinates from datafiles
        f=open(scans_path, 'rb')
        z=np.fromstring(f.read(),dtype='uint32')                           #Load what seems to be full contents of .shots file into RAM, + some overhead (20%/20%)
        f.close()
        print('Writing scans...')
        for i in range(n_blocks):
            low = i * chunk_size
            high = low + chunk_size
            block = [z[sii[jj]] for jj in range(low, high)]
            dset[low:high,2]=block
            del block
        low = n_blocks * chunk_size
        high = low + remainder
        block = [z[sii[jj]] for jj in range(low, high)]
        dset[low:high,2]=block
        layers=z.max()+1
        del z, block
        h5f.flush()
        if nuke: os.remove(scans_path)
        f=open(coords_path, 'rb')
        coords=np.fromstring(f.read(),dtype='uint32')                           #Similar for .coords file (40%/60%)
        f.close()
        cols=0
        rows=0
        print('Writing coordinates...')
        for i in range(n_blocks):
            low = i * chunk_size
            high = low + chunk_size
            xblock = [coords[sii[jj]*2] for jj in range(low, high)]
            yblock = [coords[(sii[jj]*2)+1] for jj in range(low, high)]
            if np.max(xblock) > cols: cols = np.max(xblock)
            if np.max(yblock) > rows: rows = np.max(yblock)
            dset[low:high,0]=xblock
            dset[low:high,1]=yblock
            del xblock
            del yblock
        low = n_blocks * chunk_size
        high = low + remainder
        xblock = [coords[sii[jj]*2] for jj in range(low, high)]
        yblock = [coords[(sii[jj]*2)+1] for jj in range(low, high)]
        dset[low:high,0]=xblock
        dset[low:high,1]=yblock
        del coords, xblock, yblock
        h5f.flush()
        if nuke: os.remove(coords_path)
        cols += 1
        rows += 1
        
        #Read in measurement parameters from properties plaintext file
        parms_path=os.path.join(path, file_prefix+'.properties.txt')
        f=open(parms_path, encoding='ISO-8859-1')
        sims_parms={}
        for line in f:
            words=line.split()
            sims_parms[words[0]]=float(words[2])
        f.close()
        #Read in calibration parameters
        SF=sims_parms['Context.MassScale.SF']
        K0=sims_parms['Context.MassScale.K0']
        scan_size=sims_parms['Registration.Raster.FieldOfView']*1e6

        for name,val in sims_parms.items():
            dset.attrs[name]=val

        grp.attrs['Total_counts']=counts
        grp.attrs['SF']=SF
        grp.attrs['K0']=K0
        grp.attrs['x_points']=cols
        grp.attrs['y_points']=rows
        grp.attrs['z_points']=layers

        h5f.flush()
        if nuke: os.remove(parms_path)
        return h5f

    def load_from_array(self, x,y,z,tofs, SF, K0, scan_size):
        '''
        Loads raw SIMS data numpy arrays

        Inputs
        --------
        x,y,z,tofs : 1D numpy arrays
            x,y,z coordinates and corresponding times of flight
        SF, K0 : float
            SF and K0 calibration coefficients
        scan_size : float
            Scan scize in um
        '''
        self.x=x
        self.y=y
        self.z=z
        self.tofs=tofs

        self.SF=SF
        self.K0=K0
        self.scan_size=scan_size

        self.layers=self.z.max()+1
        self.cols=self.x.max()+1
        self.rows=self.y.max()+1

        self.counts=self.tofs.size

    def tof_correction(self, new_tofs, h5f):
        '''
        Replaces times of flights by corrected data

        Inputs
        --------
        new_tofs : 1D numpy array
            Corrected times of flight
        '''
        #If this is first time we are doing correction
        if not 'Raw_data_uncorr' in h5f['Raw_data'].keys():
            dset=h5f['Raw_data'].create_dataset('Raw_data_uncorr')
            dset=h5f['Raw_data']['Raw_data']
            h5f.flush()

        #Change tof column
        h5f['Raw_data']['Raw_data'][:,3]=new_tofs
        h5f.flush()


    def select_z_range(self, z_range):
        '''
        Selects range of z coordinates used in analysis and coversion

        Inputs
        --------
        (z_strat, z_end) : tuple of ints
            indecies of start and end layers
        '''
        ii=np.where((self.z>=z_range[0]) & (self.z<z_range[1]))
        self.z=self.z[ii]
        self.z=self.z-z_range[0]

        self.x=self.x[ii]
        self.y=self.y[ii]
        self.tofs=self.tofs[ii]
        self.layers=self.z.max()+1

    def get_2d_rawdata(self):
        rawdata2d=np.zeros((self.x.size, 4),dtype='uint32')
        rawdata2d[:,0]=self.x
        rawdata2d[:,1]=self.y
        rawdata2d[:,2]=self.z
        rawdata2d[:,3]=self.tofs
        return rawdata2d

    def write_to_h5(self, h5f):
        '''
        DEPRECATED
        Writes loaded raw data to HDF5 datafile.
        
        Inputs
        --------
        h5f : str
            filename for HDF5 file
        '''
        #If previously imported data exists, delete it
        if 'Raw_data' in h5f.keys():
            del h5f['Raw_data']
            
        grp=h5f.create_group('Raw_data')

        dset=grp.create_dataset('Raw_data', (self.counts, 4), dtype='uint32',
                                chunks=(1024*1024, 1))    #Implement chunking of 1-D data
        #!!!Something about the chunking scheme here seems odd
        #!!!Consider breaking this into separate datasets for tofs and coords
        
        
        
        

        
        return
