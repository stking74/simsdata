# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:03:19 2016

@author: iv1
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, Lock
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
        self.h5f=h5f
        self.h5_path=h5f.filename
        self.name=name        
        self.h5f=h5f
        self.tof_resolution=conv_model.tof_resolution
        self.xy_bins=conv_model.xy_bins
        self.width_threshold=conv_model.width_threshold
        self.counts_threshold=conv_model.counts_threshold
        self.cores=conv_model.cores
        self.chunk_size=conv_model.chunk_size
        
        self.raw_x_points=h5f['Raw_data'].attrs['x_points']
        self.raw_y_points=h5f['Raw_data'].attrs['y_points']
        self.raw_z_points=h5f['Raw_data'].attrs['z_points']
        self.tofs_max=h5f['Raw_data'].attrs['spectra_points']
        
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
        
            
    def _peaks_detection(self):
        t0=time.time()
        print("SIMS data averaging...")
        
        counts = self.h5f['Raw_data']['Raw_data'].shape[0]        
        sum_spectrum=np.zeros(int(self.tofs_max/self.tof_resolution))
        if self.cores==1:
            tofs=self.h5f['Raw_data']['Raw_data'][:,3]
            for start_i in range(0,counts,self.chunk_size):
                chunk=self.h5f['Raw_data']['Raw_data'][start_i:(start_i+self.chunk_size),3] if (start_i+self.chunk_size)<self.tofs_max else\
                        self.h5f['Raw_data']['Raw_data'][start_i:,3]
                spectrum,b=np.histogram(chunk, int(self.tofs_max/self.tof_resolution), 
                                     (0,int(self.tofs_max/self.tof_resolution)*self.tof_resolution))
                sum_spectrum+=spectrum
                del(b)

        else:
            p_wrapped=zip(range(0,counts,self.chunk_size), 
                          itertools.repeat((self.h5_path,self.chunk_size,self.tof_resolution,self.tofs_max)))
 
            lock=Lock()
            pool=Pool(processes=self.cores,initializer=init_mp_lock, initargs=(lock,))

            mapped_results=pool.imap_unordered(chunk_spectrum, p_wrapped) 
            
            i=0
            for spectrum in mapped_results:
                sum_spectrum+=spectrum
                i+=1

            pool.close()

        f,ax=plt.subplots(nrows=2,figsize=(15,15))
        ax[0].plot(sum_spectrum)
        
        max_signal = sum_spectrum.max()
        threshold = max_signal * self.counts_threshold
        signal_ii=np.where(sum_spectrum>threshold)[0]
        ax[1].plot(sum_spectrum[signal_ii])
        
        self.sum_spectrum=sum_spectrum
        
        del(spectrum)
        del(sum_spectrum)
        
        print("SIMS averaging completed %ds"%(time.time()-t0))
        return signal_ii
        
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
                        
        signal_ii=self._peaks_detection()
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
        signal_ii=self._peaks_detection()
        
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
        signal_ii=self._peaks_detection()
        mass=((signal_ii*self.tof_resolution+self.K0+71*40)/self.SF)**2
        exclude_mass = np.array([])
                  
        self.spectra_tofs=peaks_filtering(signal_ii, mass, self.width_threshold, exclude_mass)       
        self.spectra_mass=((self.spectra_tofs*self.tof_resolution+self.K0+71*40)/self.SF)**2 
        self.spectra_len=len(self.spectra_tofs)
        
        self.create_h5_grp()
        self.h5f[self.h5_grp_name].create_dataset('Excluded_peaks',data=exclude_mass)
        
    def _plot_output(self,output):
        self.pout=output
        self.Wait=False
        
    def convert(self):
        if self.cores>1:
            self._convert_multiprocessing()
        else:
            self._convert_single_process()
        
    def _convert_single_process(self):
        t0=time.time()
        print("SIMS data conversion...")
        
        counts = self.h5f['Raw_data']['Raw_data'].shape[0]             
        
        data4d = np.zeros((self.z_points, self.x_points, self.y_points, self.spectra_len))
        
        ave_3d_map = np.zeros((self.z_points,self.x_points, self.y_points))
        ave_spectrum=np.zeros(self.spectra_len)
        chunk_no=0
        
        for start_i in range(0,counts, self.chunk_size):
            out_block=convert_chunk((start_i,(self.h5_path, self.chunk_size,self.xy_bins, 
                                              self.z_bins, self.spectra_tofs, self.tof_resolution)))
            for spectrum in out_block:
                x, y, z = spectrum[:3]
                data4d[z,x,y] += spectrum[3:]
                ave_3d_map[z,x,y] += np.sum(spectrum[3:])
                ave_spectrum += spectrum[3:]
            chunk_no+=1
            print('%d chunks of %d are processed'%(chunk_no, int(counts/self.chunk_size)))
            
                           
        print("SIMS saving converted data...")
        self._save_converted_data(data4d.reshape((-1, self.spectra_len)), ave_spectrum,ave_3d_map)
        print("Conversion complete. %d sec"%(time.time()-t0))  
        
    
    def _convert_multiprocessing(self, shift_corr=False):
        '''
        Start data conversion process
        
        Input:
        --------
        cores : int
            Number of used CPU cores
        '''

        t0=time.time()
        print("SIMS data conversion...")
                
        counts = self.h5f['Raw_data']['Raw_data'].shape[0]             
               
        p_wrapped=zip(range(0,counts,self.chunk_size), 
                   itertools.repeat((self.h5_path,self.chunk_size,self.xy_bins, self.z_bins, 
                                     self.spectra_tofs, self.tof_resolution)))       
        lock=Lock()
        pool=Pool(processes=self.cores,initializer=init_mp_lock, initargs=(lock,))
        mapped_results=pool.imap_unordered(convert_chunk, p_wrapped)         
               
        ave_spectrum=np.zeros(self.spectra_len)
        
        ave_3d_map = np.zeros((self.z_points,self.x_points, self.y_points))
        data4d = np.zeros((self.z_points, self.x_points, self.y_points, self.spectra_len))
        
        chunk_no=0
        for out_block in mapped_results:
            for spectrum in out_block:
                x, y, z = spectrum[:3]               
                data4d[z,x,y] += spectrum[3:]
                ave_3d_map[z,x,y] += np.sum(spectrum[3:])
                ave_spectrum += spectrum[3:]
                
            chunk_no+=1
            print('%d chunks of %d are processed'%(chunk_no, int(counts/self.chunk_size)))

        pool.close()
        print("SIMS saving converted data...")
        self._save_converted_data(data4d.reshape((-1, self.spectra_len)), ave_spectrum,ave_3d_map)
        print("Conversion complete. %d sec"%(time.time()-t0))  
 
    def _save_converted_data(self,data2d, ave_spectrum, total_3d_map):
        self.h5f[self.h5_grp_name].create_dataset('Data_full',data=data2d)       
        grp=self.h5f[self.h5_grp_name].create_group('Averaged_data')
        grp.create_dataset('Ave_spectrum',data=ave_spectrum)
        grp.create_dataset('Total_3d_map',data=total_3d_map)
        self.h5f.flush()      
       
    
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
        
        
