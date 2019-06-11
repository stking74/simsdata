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

    def load_from_file(self, path, file_prefix, chunk_order=None, unit_size=None, nuke=False):
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
        h5f = h5py.File(h5_path, 'w',libver='latest',swmr=True)
        
        counts=int(os.path.getsize(scans_path)/4)
        if 'Raw_data' in h5f.keys():
            del h5f['Raw_data']
        grp=h5f.create_group('Raw_data')        
        
        if not chunk_order or not unit_size:
            chunk_size=counts
        else:
            chunk_size=int(2**(10*chunk_order)*unit_size/4)                     #Number of values in one chunk
        
        #Dynamically resize chunks for exceptionally small datasets
        while True:
            try: 
                dset=grp.create_dataset('Raw_data', (counts, 4), dtype='uint32',chunks=(chunk_size,1))
                break
            except ValueError: chunk_size = int(chunk_size / 2)

        f_tofs=open(tofs_path, 'rb')
        f_scans=open(scans_path, 'rb')
        f_coords=open(coords_path, 'rb')
        
        x_max,y_max, z_max, tofs_max=(0,0,0,0)
        
        tofs_chunk=np.zeros(chunk_size)
        print('Writing tofs...')
        i=0
        while len(tofs_chunk)==chunk_size:
            tofs_chunk=np.frombuffer(f_tofs.read(chunk_size*4),dtype='uint32')
            dset[i*chunk_size:(i+1)*chunk_size,3]=tofs_chunk
            tofs_max=tofs_chunk.max() if tofs_chunk.max()>tofs_max else tofs_max
            print('Chunk %d completed'%(i))        
            i+=1
        del(tofs_chunk)
        h5f.flush()
        f_tofs.close()
        if nuke: os.remove(tofs_path)
        scans_chunk=np.zeros(chunk_size)
        print('Writing scans...')
        i=0
        while len(scans_chunk)==chunk_size:
            scans_chunk=np.frombuffer(f_scans.read(chunk_size*4),dtype='uint32')
            dset[i*chunk_size:(i+1)*chunk_size,2]=scans_chunk
            z_max=scans_chunk.max() if scans_chunk.max()>z_max else z_max
            print('Chunk %d completed'%(i))        
            i+=1
        del(scans_chunk)
        h5f.flush()
        f_scans.close()
        if nuke: os.remove(scans_path)
        coords_chunk=np.zeros(chunk_size*2)
        print('Writing coordinates...')
        i=0
        while len(coords_chunk)==chunk_size*2:
            coords_chunk=np.frombuffer(f_coords.read(chunk_size*8),dtype='uint32')
            dset[i*chunk_size:(i+1)*chunk_size,0]=coords_chunk[::2]
            dset[i*chunk_size:(i+1)*chunk_size,1]=coords_chunk[1::2]
            x_max=coords_chunk[::2].max() if coords_chunk[::2].max()>x_max else x_max
            y_max=coords_chunk[1::2].max() if coords_chunk[1::2].max()>y_max else y_max
            print('Chunk %d completed'%(i))        
            i+=1
        del(coords_chunk)
        h5f.flush()
        f_coords.close()
        if nuke: os.remove(coords_path)
            
        #Read in measurement parameters from properties plaintext file
        parms_path=os.path.join(path, file_prefix+'.properties.txt')
        f=open(parms_path, encoding='ISO-8859-1')
        sims_parms={}
        for line in f:
            try:
                words=line.split()
                sims_parms[words[0]]=float(words[2])
            except ValueError:
                line = line[2:]
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
        grp.attrs['x_points']=x_max+1
        grp.attrs['y_points']=y_max+1
        grp.attrs['z_points']=z_max+1
        grp.attrs['spectra_points']=tofs_max+1

        h5f.flush()
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
        #!!!Consider breaking this into separate datasets for tofs and coords
        
        
        
        

        
        return
