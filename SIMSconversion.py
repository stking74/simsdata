# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:03:19 2016

@author: iv1
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import h5py

from .SIMSmodel import SIMSmodel
from .SIMSutils import *
from .SIMSpeakbrowser import SIMSPeakBrowser

import time

class SIMSconversion(object):
    '''
    Class performing conviersion of SIMS data into numpy array
    '''
    def __init__(self, name, h5f, conv_model):
        '''
        Class constructor
        
        Inputs:
        --------
        raw_data : SIMSrawdata
            GRD SIMS data
        conv_model : SIMSmodel
            Conversion model
        '''
        print('Initializing converter...')
        self.name=name        
        
        self.h5f=h5f
        self.tof_resolution=conv_model.tof_resolution
        self.xy_bins=conv_model.xy_bins
        self.width_threshold=conv_model.width_threshold
        self.counts_threshold=conv_model.counts_threshold
        
        
        self.raw_x_points=h5f['Raw_data'].attrs['x_points']
        self.raw_y_points=h5f['Raw_data'].attrs['y_points']
        self.raw_z_points=h5f['Raw_data'].attrs['z_points']
        
        self.SF=h5f['Raw_data'].attrs['SF']
        self.K0=h5f['Raw_data'].attrs['K0']
        
        self.x_points=int(self.raw_x_points/self.xy_bins)
        self.y_points=int(self.raw_y_points/self.xy_bins)
        
        self.save_mode=conv_model.save_mode
        
        if conv_model.z_points:
            self.z_points=conv_model.z_points
            self.z_bins=int(self.raw_z_points/self.z_points)
        else:
            self.z_bins=conv_model.z_bins
            self.z_points=int(self.raw_z_points/self.z_bins)
        print('Converter initialized')
        
            

        
    def select_peak(self, mass_target, shift_corr=False):
        '''
        Selects single peak for data conversion
        
        Input:
        --------
        mass_target : float
            Mass corresponding to selected peak
        '''
        
        if shift_corr and 'Raw_data_uncorr' in self.h5f['Raw_data']:
            dataset='Raw_data_uncorr'
        else:
            dataset='Raw_data'
                        
        signal_ii,counts=peaks_detection(self.h5f['Raw_data'][dataset][:,3], self.tof_resolution, self.counts_threshold)
        mass=((signal_ii*self.tof_resolution+self.K0+71*40)/self.SF)**2           
        
        if not mass_target:
            f,ax=plt.subplots(2,1)
            plt.ion()
            plt.show()
            
            self.Wait=True
            PBrowser=SIMSPeakBrowser(f,ax,signal_ii,mass,counts,self._plot_output)
            f.canvas.mpl_connect('button_press_event', PBrowser.onclick)
            print('Waiting for data from plot...')            
            while self.Wait: 
                plt.pause(0.1)
            mass_target=self.pout[0]
        
        peak_ii=find_peak(signal_ii, mass, mass_target)
        
        self.spectra_tofs=signal_ii[peak_ii]
        self.spectra_mass=mass[peak_ii]
        self.spectra_len=len(self.spectra_tofs)
        
        self.create_h5_grp()
        self.h5f[self.h5_grp_name].create_dataset('Single_peak',data=mass_target)
        
        
        
    def select_few_peaks(self, mass_targets):
        '''
        Selects a set of peaks for data conversion
        
        Input:
        --------
        mass_targets : list
            Masses of corresponding peaks
        '''
        
        signal_ii,counts=peaks_detection(self.h5f['Raw_data']['Raw_data'][:,3], self.tof_resolution, self.counts_threshold)
        mass=((signal_ii*self.tof_resolution+self.K0+71*40)/self.SF)**2         

        self.spectra_tofs=np.empty(0)
        self.spectra_mass=np.empty(0)
        
        for mass_target in mass_targets:
            peak_ii=find_peak(signal_ii, mass, mass_target)

            self.spectra_tofs=np.append(self.spectra_tofs,signal_ii[peak_ii])
            self.spectra_mass=np.append(self.spectra_mass,mass[peak_ii])
            self.spectra_len=len(self.spectra_tofs)
                    
        self.create_h5_grp()
        self.h5f[self.h5_grp_name].create_dataset('Selected_peaks',data=mass_target)
        
    def select_all_peaks(self,exclude_mass=None):
        '''
        Selects all peaks exluding specified for data conversion
        
        Input:
        --------
        exclude_mass : list
            Masses of peaks to be excluded
        '''
        print('Detecting peaks...')
        signal_ii,counts=peaks_detection(self.h5f['Raw_data']['Raw_data'][:,3], self.tof_resolution, self.counts_threshold)
        mass=((signal_ii*self.tof_resolution+self.K0+71*40)/self.SF)**2
        print('Please select peaks to exclude')
        exclude_mass = np.array([])
		
#        if not exclude_mass:
#            f,ax=plt.subplots(2,1)
#            plt.ion()
#            plt.show()
#           
#            self.Wait=True
#            PBrowser=SIMSPeakBrowser(f,ax,signal_ii,mass,counts,self._plot_output)
#            f.canvas.mpl_connect('button_press_event', PBrowser.onclick)
#            print('Waiting for data from plot...')            
#            while self.Wait: 
#                plt.pause(0.1)
#            exclude_mass=self.pout
#            print(exclude_mass)
                   
        self.spectra_tofs=peaks_filtering(signal_ii, mass, self.width_threshold, exclude_mass)       
        self.spectra_mass=((self.spectra_tofs*self.tof_resolution+self.K0+71*40)/self.SF)**2 
        self.spectra_len=len(self.spectra_tofs)
        
        self.create_h5_grp()
        self.h5f[self.h5_grp_name].create_dataset('Excluded_peaks',data=exclude_mass)
        
    def _plot_output(self,output):
        self.pout=output
        self.Wait=False
        
    def convert_single_process(self):
        print("SIMS data preparation...")
        
        parms=(self.x_points, self.y_points, self.xy_bins, 
                                  self.spectra_tofs, self.tof_resolution)
        
        print("SIMS data conversion started...")        
        data2d,count=convert_data3d_track(self.h5f,parms)
        print("SIMS converted data saving...")        
        
        self.h5f[self.h5_grp_name].create_dataset('Data_full',data=data2d)
        self.h5f.flush()
        #self.h5f[self.h5_grp_name]['Data_full']=data2d
      
        grp=self.h5f[self.h5_grp_name].create_group('Averaged_data')
        grp.create_dataset('Ave_spectrum',data=data2d.mean(axis=0))
        grp.create_dataset('Total_3d_map',data=data2d.reshape((self.x_points,self.y_points,-1)).sum(axis=2).reshape((1,self.x_points,self.y_points)))
        self.h5f.flush()      
    
    def convert_multiprocessing(self, cores=5, shift_corr=False):
        '''
        Start multiprocessing conversion process
        
        Input:
        --------
        cores : int
            Number of used CPU cores
        '''
        if shift_corr and 'Raw_data_uncorr' in self.h5f['Raw_data']:
            dataset='Raw_data_uncorr'
        else:
            dataset='Raw_data'
        
        counts = self.h5f['Raw_data'].shape[0]
        chunk_size = 10000
        n_chunks = counts // chunk_size
        remainder = chunk_size % n_chunks
        t0=time.time()
        
        p_parms=itertools.repeat((self.x_points, self.y_points, self.xy_bins, 
                                  self.spectra_tofs, self.tof_resolution))
        
        p_wrapped=[(self.h5f, i*chunk_size, chunk_size, (self.xy_bins, self.z_bins, self.spectra_tofs, self.tof_resolution)) for i in range(n_chunks)]
        p_wrapped.append((self.h5f, (i+1)*chunk_size, remainder, (self.xy_bins, self.z_bins, self.spectra_tofs, self.tof_resolution)))
        print("SIMS data preparation complete. %d sec"%(time.time()-t0))
        t0=time.time()
        print("SIMS data conversion...")
                          
        pool=Pool(processes=cores)
        mapped_results=pool.imap(convert_chunk, p_wrapped, chunksize=1) 
        pool.close() 
        
        #data4d=np.zeros((self.z_points,self.x_points,self.y_points,self.spectra_len))
        
        layers=self.z_points if self.save_mode=='bin' else self.raw_z_points
        layer_bins=self.z_bins if self.save_mode=='bin' else 1
        
        ave_spectrum=np.zeros(self.spectra_len)
        
        n_spectra=0 
        ave_3d_map = np.zeros(shape=(self.x_points, self.y_points))
        data3d = np.zeros(shape=self.x_points, self.y_points, self.z_points, len(self.spectra_len))
        for out_block in mapped_results:
            for spectrum in out_block:
                x, y, z = tuple(spectrum[:3])
                spectrum = spectrum[3:]
                data3d[x,y,z] += spectrum
                ave_3d_map[x, y] += np.sum(spectrum)
                ave_spectrum += spectrum
                n_spectra += 1
         
        self.h5f[self.h5_grp_name].create_dataset('Data_full',data=data3d)
        self.h5f.flush()
        
        grp=self.h5f[self.h5_grp_name].create_group('Averaged_data')
        grp.create_dataset('Ave_spectrum',data=ave_spectrum/n_spectra)
        grp.create_dataset('Total_3d_map',data=ave_3d_map)
        self.h5f.flush()      

        print("Conversion complete. %d sec"%(time.time()-t0))  
 
    def create_h5_grp(self):        
        m=1
        for name in self.h5f.keys():
            parts=name.split('_')
            if parts[0]=='Converted' and int(parts[1])>=m:
                m=int(parts[1])+1
        grp_name='Converted_'+str(m).zfill(3)
        self.h5_grp_name=grp_name
                       
        grp=self.h5f.create_group(grp_name)        
                       
        grp.attrs['tof_resolution']=self.tof_resolution
        grp.attrs['xy_bins']=self.xy_bins
        grp.attrs['z_bins']=self.z_bins
        grp.attrs['counts_threshold']=self.counts_threshold
        grp.attrs['width_threshold']=self.width_threshold
        
        grp.attrs['x_points']=self.x_points
        grp.attrs['y_points']=self.y_points
        grp.attrs['z_points']=self.z_points
        
        grp.attrs['name']=self.name
        
        grp.attrs['save_mode']=self.save_mode
        
        dset_tofs=grp.create_dataset('spectra_tofs',data=self.spectra_tofs)
        dset_mass=grp.create_dataset('spectra_mass',data=self.spectra_mass)
        dset_mass=grp.create_dataset('spectra_len',data=self.spectra_len)
        
        self.h5f.flush()
        
        
