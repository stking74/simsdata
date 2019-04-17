from __future__ import division # int/int = float

import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import h5py

from SIMSutils import *

import time

from sklearn.decomposition import NMF, PCA
from scipy.ndimage import gaussian_filter

from os import path, listdir, remove # File Path formatting
from warnings import warn
from ..Translator import Translator
from ..Utils import interpretFreq, makePositionMat, getPositionSlicing, generateDummyMainParms, getH5DsetRefs
from ..BEutils import parmsToDict
from ...SPMnode import AFMDataGroup, AFMDataset # The building blocks for defining heirarchical storage in the H5 file
from ...ioHDF5 import ioHDF5 # Now the translator is responsible for writing the data.

from PySPM.io.Translators.SIMS.SIMSrawdata import SIMSrawdata
from PySPM.io.Translators.SIMS.SIMSdata import SIMSdata
from PySPM.io.Translators.SIMS.SIMSmodel import SIMSmodel

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
        signal_ii,counts=peaks_detection(self.h5f['Raw_data']['Raw_data'][:,3], self.tof_resolution, self.counts_threshold)
        mass=((signal_ii*self.tof_resolution+self.K0+71*40)/self.SF)**2
        
        if not exclude_mass:
            f,ax=plt.subplots(2,1)
            plt.ion()
            plt.show()
            
            self.Wait=True
            PBrowser=SIMSPeakBrowser(f,ax,signal_ii,mass,counts,self._plot_output)
            f.canvas.mpl_connect('button_press_event', PBrowser.onclick)
            print('Waiting for data from plot...')            
            while self.Wait: 
                plt.pause(0.1)
            exclude_mass=self.pout
            print(exclude_mass)
                   
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
        x_data=self.h5f['Raw_data']['Raw_data'][:,0]
        y_data=self.h5f['Raw_data']['Raw_data'][:,1]
        tof_data=self.h5f['Raw_data']['Raw_data'][:,3]
        
        parms=(self.x_points, self.y_points, self.xy_bins, 
                                  self.spectra_tofs, self.tof_resolution)
        
        print("SIMS data conversion started...")        
        data2d,count=convert_data3d_track((x_data,y_data,tof_data,parms))
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
            
        t0=time.time()
        print("SIMS data preparation...")
        p_x=[]
        p_y=[]
        p_tof=[]
        
        p_parms=itertools.repeat((self.x_points, self.y_points, self.xy_bins, 
                                  self.spectra_tofs, self.tof_resolution))

        z_data=self.h5f['Raw_data'][dataset][:,2]
        x_data=self.h5f['Raw_data']['Raw_data'][:,0]
        y_data=self.h5f['Raw_data']['Raw_data'][:,1]
        tof_data=self.h5f['Raw_data']['Raw_data'][:,3]
#             
        i=0
        layer=0
        while layer<self.raw_z_points:
            i0=i
            i=np.searchsorted(z_data[i0:],layer+1)+i0
            #print(i)

            p_x+=[x_data[i0:i]]
            p_y+=[y_data[i0:i]]
            p_tof+=[tof_data[i0:i]]
            #print('Slicing done!')
                       
#            p_x+=[self.h5f['Raw_data'][dataset][i0:i,0]]
#            p_y+=[self.h5f['Raw_data'][dataset][i0:i,1]]
#            p_tof+=[self.h5f['Raw_data'][dataset][i0:i,3]]
            
            layer+=1
            print('Layer %d done'%(layer))
            
        p_wrapped=zip(p_x,p_y,p_tof,p_parms)            
        print("SIMS data preparation complete. %d sec"%(time.time()-t0))
        t0=time.time()
        print("SIMS data conversion...")        
        
                          
        pool=Pool(processes=cores)
        mapped_results=pool.imap(convert_data3d, p_wrapped, chunksize=1) 
        pool.close() 
        
      #  data4d=np.zeros((self.z_points,self.x_points,self.y_points,self.spectra_len))
        
        layers=self.z_points if self.save_mode=='bin' else self.raw_z_points
        layer_bins=self.z_bins if self.save_mode=='bin' else 1
        
        self.h5f[self.h5_grp_name].create_dataset('Data_full',(layers*self.x_points*self.y_points, self.spectra_len))
        
        ave_spectrum=np.zeros(self.spectra_len)
        ave_3d_map=np.zeros((self.raw_z_points,self.x_points,self.y_points))
        
        s=0
        bin_ii=0
        complete=0
        total_counts=0 

        data_slice=np.zeros((self.x_points*self.y_points, self.spectra_len))
        for (data2d,count) in mapped_results:
           bin_ii+=1
           data_slice+=data2d
           if bin_ii==layer_bins:
               ii_start=self.x_points*self.y_points*int(s/layer_bins)
               ii_end = self.x_points*self.y_points*(int(s/layer_bins)+1)
               self.h5f[self.h5_grp_name]['Data_full'][ii_start:ii_end,:]=data_slice              
               self.h5f.flush()
               #Reset vars
               data_slice[...]=0
               bin_ii=0
           
           total_counts+=count
           print('Layer %d. %d counts'%(s, count))
           
           ave_spectrum+=data2d.sum(axis=0)
           ave_3d_map[s,:,:]=data2d.sum(axis=1).reshape((self.x_points,self.y_points))
           
           s+=1                      
           if s*100/self.raw_z_points-complete>=10:
               complete+=10
               print("%d%% complete"%(complete))
               
        del(p_wrapped)
        del(p_x)        
        del(p_y)        
        del(p_tof)   
        
        grp=self.h5f[self.h5_grp_name].create_group('Averaged_data')
        grp.create_dataset('Ave_spectrum',data=ave_spectrum)
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

class SIMSdata(object):
    '''
    Class stroing and processing converted SIMS data
    '''

    def __init__(self, h_ex=1.5e-3, U_ex=2e3):
        self.h_ex = h_ex
        self.U_ex = U_ex

    def close(self):
        self.h5f.flush()
        self.h5f.close()

    def load_raw(self, path, file_prefix):
        raw_data = SIMSrawdata()
        raw_data.load_from_file(path, file_prefix)

        h5_path = path+'\\'+file_prefix+'.h5'

        self.h5f = h5py.File(h5_path)
        raw_data.write_to_h5(self.h5f)

    def load_h5(self, h5_path):
        self.h5f = h5py.File(h5_path)

    def convert_data(self, name, conv_model, cores=5, **kwargs):
        '''
        Starts data conversion process

        Input:
        --------
        cores : int
            number of cores used in conversion process (default cores=2)
        '''
        #Shift correction if required
        self.shift_correction(conv_model, cores=cores)

        #Data conversion
        conv_data = SIMSconversion(name, self.h5f, conv_model)

        if conv_model.conv_all:
            conv_data.select_all_peaks(conv_model.exclude_mass)
        else:
            conv_data.select_few_peaks(conv_model.peaks_mass)

        if 'single_process' not in kwargs.keys():
            print('Single_process')
            conv_data.convert_single_process()
            #conv_data.convert_multiprocessing(cores=cores)
        else:
            print('Single_process')
            conv_data.convert_single_process()

        self.spectra_tofs = conv_data.spectra_tofs
        self.spectra_mass = conv_data.spectra_mass
        self.spectra_len = self.spectra_tofs.size

        self.x_points = conv_data.x_points
        self.y_points = conv_data.y_points
        self.z_points = conv_data.z_points

        print("h5_grp_name:", conv_data.h5_grp_name)

        self.data = self.h5f[conv_data.h5_grp_name]['Data_full']

    def load_converted_data(self, conv_no):
        conv_name = 'Converted_'+str(conv_no).zfill(3)
        if not conv_name in self.h5f.keys():
            print('Converted data not found')
        else:
            self.data = self.h5f[conv_name]['Data_full']
            self.spectra_tofs = self.h5f[conv_name]['spectra_tofs'][:]
            self.spectra_mass = self.h5f[conv_name]['spectra_mass'][:]
            self.spectra_len = self.h5f[conv_name]['spectra_len'].value

            self.x_points = self.h5f[conv_name].attrs['x_points']
            self.y_points = self.h5f[conv_name].attrs['y_points']
            self.z_points = self.h5f[conv_name].attrs['z_points']
            self.conv_name = conv_name

    def load_processed_data(self, name):
        self.analysis_name = name

    def list_converted_data(self):
        s = 0
        for name in self.h5f.keys():
            arr = name.split('_')
            if arr[0] == 'Converted':
                print('Dataset %s, name=%s'%(name, self.h5f[name].attrs['name']))
                for subname in self.h5f[name].keys():
                    if isinstance(self.h5f[name][subname], h5py.Group):
                        print('\tSubfolder %s'%(subname))
                s += 1
        if not s:
            print('No converted datasets found')


    def shift_correction(self, conv_model, cores):
        '''
        Performs ToF correction with data automticaly rewriten in rawdata

        Input:
        --------
        cores : int
            Number of the cores used in multiprocessing (default cores=2)
        '''
        if conv_model.shift_corr:
            print("Shift correction...")
            conv_data = SIMSconversion('Shift correction', self.h5f, conv_model)
            conv_data.select_peak(conv_model.corr_peak, shift_corr=True)
            conv_data.convert_multiprocessing(cores=cores, shift_corr=True)

            #Calaculation of the peak position
            data2d = self.h5f[conv_data.h5_grp_name]['Data_full'][:, :]

            count_sum = data2d.sum(axis=1)
            mass_centers = np.sum(data2d*conv_data.spectra_tofs, axis=1)/count_sum
            mass_centers[np.where(count_sum < 10)] = np.nan

            max_pos = np.nanmax(mass_centers)
            layers = conv_data.raw_z_points if conv_data.save_mode == 'all' else conv_data.z_points
            #Correction coefficient
            D = max_pos/mass_centers.reshape((layers, conv_data.x_points, conv_data.y_points))


            f, ax = plt.subplots()
            ax.imshow(D.sum(axis=0))

            self.h5f['Raw_data'].create_dataset('D', data=D)
            self.h5f['Raw_data']['D'].attrs['Correction mode'] = 'Proportional using centroids'
            self.h5f['Raw_data']['D'].attrs['Max_peak_position'] = max_pos
            self.h5f['Raw_data']['D'].attrs['Peak_mass'] = \
                conv_data.spectra_mass[np.searchsorted(conv_data.spectra_tofs, max_pos)]
            self.h5f['Raw_data']['D'].attrs['tof_resolution'] = conv_data.tof_resolution
            self.h5f.flush()

            #Shift correction
            x_data = self.h5f['Raw_data']['Raw_data'][:, 0]
            y_data = self.h5f['Raw_data']['Raw_data'][:, 1]
            z_data = self.h5f['Raw_data']['Raw_data'][:, 2]
            tof_data = self.h5f['Raw_data']['Raw_data'][:, 3]

            z_bins = 1 if conv_data.save_mode == 'all' else conv_data.z_bins

            p_wrapped = zip(np.array_split(x_data, cores), np.array_split(y_data, cores),
                            np.array_split(z_data, cores), np.array_split(tof_data, cores),
                            repeat((D, conv_model.xy_bins, z_bins)))
            print('ToF shift correction...')
            t0 = time.time()
            pool = Pool(processes=cores)
            mapped_results = pool.imap(correct_shift, p_wrapped, chunksize=1)
            pool.close()
            tofs_corr = np.empty(0, dtype='uint32')

            complete = 0
            s = 0
            for t in mapped_results:
                tofs_corr = np.append(tofs_corr, t)

                s += 1
                if s*100/cores-complete >= 10:
                    complete += 10
                    print("%d%% complete"%(complete))

            del p_wrapped

            if not 'Raw_data_uncorr' in self.h5f['Raw_data'].keys():
                self.h5f['Raw_data'].create_dataset('Raw_data_uncorr',
                                                    data=self.h5f['Raw_data']['Raw_data'])
                self.h5f.flush()

            #Change tof column
            self.h5f['Raw_data']['Raw_data'][:, 3] = tofs_corr
            self.h5f.flush()

            print("Shift correction complete. Time %d sec"%(time.time()-t0))

    def plot_ave_data(self):
        ave_spectrum = self.h5f[self.conv_name]['Averaged_data']['Ave_spectrum'][:]
        ave_3dmap = self.h5f[self.conv_name]['Averaged_data']['Total_3d_map'][:, :]

        f, ax = plt.subplots(2, 1)
        f2, ax2 = plt.subplots(1, 3)

        ax[0].plot(ave_spectrum, linewidth=1.5, color='b')
        ax[0].set_title('Averaged mass spectrum')
        tof_ext, mass_ext, ave_ext = self.get_extended_spectra(ave_spectrum)
        ax[1].plot(mass_ext, ave_ext, linewidth=1.5, color='r')

        ax2[0].imshow(ave_3dmap.sum(axis=0), cmap=plt.cm.jet)
        ax2[0].set_title('XY map')
        ax2[1].imshow(ave_3dmap.sum(axis=2), cmap=plt.cm.jet)
        ax2[1].set_title('ZX map')
        ax2[2].imshow(ave_3dmap.sum(axis=1), cmap=plt.cm.jet)
        ax2[2].set_title('ZY map')

    def get_extended_spectra(self, y_data):
        tof_ext = np.empty(0)
        y_data_ext = np.empty(0)

        i = 0
        while i < self.spectra_len:
            i0 = i
            while i < self.spectra_len-1 and self.spectra_tofs[i+1] == self.spectra_tofs[i]+1:
                i += 1
            tof_ext = np.append(tof_ext, [self.spectra_tofs[i0]-1])
            tof_ext = np.append(tof_ext, self.spectra_tofs[i0:i+1])
            tof_ext = np.append(tof_ext, [self.spectra_tofs[i]+1])
            y_data_ext = np.append(y_data_ext, [0])
            y_data_ext = np.append(y_data_ext, y_data[i0:i+1])
            y_data_ext = np.append(y_data_ext, [0])
            i += 1
        mass_ext = self.tof_to_mass(tof_ext)

        return tof_ext, mass_ext, y_data_ext

    def tof_to_mass(self, tof):
        SF = self.h5f['Raw_data'].attrs['SF']
        K0 = self.h5f['Raw_data'].attrs['K0']
        tof_resolution = self.h5f[self.conv_name].attrs['tof_resolution']

        return ((tof*tof_resolution+K0+71*40)/SF)**2

    def NMF(self, comp_num, **kwargs):
        #load and prepare data
        print('NMF data preparation...')
        data4d = self._data_preparation(**kwargs)

        (z_points, x_points, y_points, s_points) = data4d.shape
        data = data4d.reshape((-1, s_points))

        print('NMF claculation...')
        model = NMF(n_components=comp_num, init='random', random_state=0)
        model.fit(data)
        spectra = model.components_

        print('NMF data transformation...')
        maps = model.transform(data).reshape((z_points, x_points, y_points, -1))

        #Normalizing loading to 1
        coef = maps.reshape((-1, comp_num)).max(axis=0)
        maps /= coef
        spectra = (spectra.T*coef).T

        #Saving data
        m = 1
        for name in self.h5f[self.conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'NMF' and int(parts[1]) >= m:
                m = int(parts[1])+1

        grp_name = 'NMF_'+str(m).zfill(3)
        grp = self.h5f[self.conv_name].create_group(grp_name)
        grp.attrs['Number_of_components'] = comp_num
        grp.attrs['Mode'] = 'Standard NMF with random initial values'

        if 'spatial_range' in kwargs.keys():
            grp.attrs['x_points'] = x_points
            grp.attrs['y_points'] = y_points
            grp.attrs['z_points'] = z_points
            grp.attrs['s_points'] = s_points

        grp.create_dataset('Maps', data=maps)
        grp.create_dataset('Endmembers', data=spectra)

        self.h5f.flush()
        self.analysis_name = grp_name

        #Plotting data
        if 'plot' in kwargs.keys() and kwargs['plot']:
            self.plot_endmembers()
            self.plot_loading_maps()

    def plot_endmembers(self, mode='split', **kwargs):
        '''
        Plotting modes:
        'split': plot in sepparate axex
        'merge': plot in same axis delta is defined as a list or value in kwargs
        '''
        spectra = self.h5f[self.conv_name][self.analysis_name]['Endmembers'][:, :]

        spectra_ext = np.zeros((spectra.shape[0], self.spectra_len))
        spectra_ext[:, :] = spectra
        spectra = spectra_ext

        comp_num = spectra.shape[0]

        if mode == 'split':
            f, ax = plt.subplots(comp_num, 1)
            for i in range(comp_num):
                tof, mass, data = self.get_extended_spectra(spectra[i, :])
                ax[i].plot(mass, data, linewidth=1.5, color=plt.cm.jet(int(i*255/(comp_num-1))))
        elif mode == 'merge':
            if 'delta' in kwargs.keys():
                delta = kwargs['delta']
            else:
                delta = 0
            f, ax = plt.subplots()
            for i in range(comp_num):
                tof, mass, data = self.get_extended_spectra(spectra[i, :])
                ax.plot(mass, data+delta*i, linewidth=1.5,
                        color=plt.cm.jet(int(i*255/(comp_num-1))))

    def plot_loading_maps(self, mode='cross', **kwargs):
        '''
        Plotting modes:
        'cross' - plots crossections XY, ZX, ZY in point, specified in kwarg pos
        'tomo' - plots crossection along one direction (dir), positions specified in itterable pos
        Kwargs:

        '''
        maps = self.h5f[self.conv_name][self.analysis_name]['Maps'][:, :, :, :]
        if 'filtering' in kwargs.keys() and kwargs['filtering'] == 'Gauss':
            for i in range(maps.shape[3]):
                maps[:, :, :, i] = gaussian_filter(maps[:, :, :, i],
                                                   sigma=kwargs['sigma'] if 'sigma' in kwargs.keys() else 1.0)

        if mode == 'cross':
            self._plot_maps_cross(maps, **kwargs)
        elif mode == 'tomo':
            self._plot_maps_tomo(maps, **kwargs)

    def _plot_maps_cross(self, maps, **kwargs):
        if 'x_points' in self.h5f[self.conv_name][self.analysis_name].attrs.keys():
            x_points = self.h5f[self.conv_name][self.analysis_name].attrs['x_points']
            y_points = self.h5f[self.conv_name][self.analysis_name].attrs['y_points']
            z_points = self.h5f[self.conv_name][self.analysis_name].attrs['z_points']
        else:
            x_points = self.x_points
            y_points = self.y_points
            z_points = self.z_points

        comp_num = maps.shape[3]
        if 'pos' in kwargs.keys():
            pos = kwargs['pos']
        else:
            pos = (int(x_points/2), int(y_points/2), int(z_points/2))

        f, ax = plt.subplots(comp_num, 3)
        for i in range(comp_num):
            ax[i, 0].imshow(maps[pos[2], :, :, i], cmap=plt.cm.jet)
            ax[i, 1].imshow(maps[:, :, pos[1], i], cmap=plt.cm.jet)
            ax[i, 2].imshow(maps[:, pos[0], :, i], cmap=plt.cm.jet)

    def _plot_maps_tomo(self, maps, **kwargs):
        if 'x_points' in self.h5f[self.conv_name][self.analysis_name].attrs.keys():
            x_points = self.h5f[self.conv_name][self.analysis_name].attrs['x_points']
            y_points = self.h5f[self.conv_name][self.analysis_name].attrs['y_points']
            z_points = self.h5f[self.conv_name][self.analysis_name].attrs['z_points']
        else:
            x_points = self.x_points
            y_points = self.y_points
            z_points = self.z_points

        comp_num = maps.shape[3]
        if 'pos' in kwargs.keys():
            pos = kwargs['pos']
        else:
            pos = range(0, z_points, 5)

        print(pos)

        if 'direction' in kwargs.keys():
            direction = kwargs['direction']
        else:
            direction = 'XY'

        f, ax = plt.subplots(comp_num, len(pos))
        f2, ax2 = plt.subplots(comp_num, 2)

        imh = [None]*comp_num*len(pos)
        for i in range(comp_num):
            if direction == 'XY':
                ax2[i, 0].imshow(maps[:, :, :, i].sum(axis=2), cmap=plt.cm.jet)
                ax2[i, 0].set_title('ZX cross-section')
                ax2[i, 1].imshow(maps[:, :, :, i].sum(axis=1), cmap=plt.cm.jet)
                ax2[i, 1].set_title('ZY cross-section')
            elif direction == 'ZX':
                ax2[i, 0].imshow(maps[:, :, :, i].sum(axis=0), cmap=plt.cm.jet)
                ax2[i, 0].set_title('XY cross-section')
                ax2[i, 1].imshow(maps[:, :, :, i].sum(axis=1), cmap=plt.cm.jet)
                ax2[i, 1].set_title('ZY cross-section')
            elif direction == 'ZY':
                ax2[i, 0].imshow(maps[:, :, :, i].sum(axis=0), cmap=plt.cm.jet)
                ax2[i, 0].set_title('XY cross-section')
                ax2[i, 1].imshow(maps[:, :, :, i].sum(axis=2), cmap=plt.cm.jet)
                ax2[i, 1].set_title('ZX cross-section')

            for pi in range(len(pos)):
                if direction == 'XY':
                    imh[i*len(pos)+pi] = ax[i, pi].imshow(maps[pos[pi], :, :, i], cmap=plt.cm.jet)
                    ax[i, pi].set_title('Z pos = %d'%(pos[pi]))
                    ax2[i, 0].plot([0, x_points], [pos[pi], pos[pi]], 'r--', linewidth=1.5)
                    ax2[i, 1].plot([0, y_points], [pos[pi], pos[pi]], 'r--', linewidth=1.5)
                elif direction == 'ZX':
                    imh[i*len(pos)+pi] = ax[i, pi].imshow(maps[:, pos[pi], :, i], cmap=plt.cm.jet)
                    ax[i, pi].set_title('Y pos = %d'%(pos[pi]))
                    ax2[i, 0].plot([0, x_points], [pos[pi], pos[pi]], 'r--', linewidth=1.5)
                    ax2[i, 1].plot([pos[pi], pos[pi]], [0, z_points], 'r--', linewidth=1.5)
                elif direction == 'ZY':
                    imh[i*len(pos)+pi] = ax[i, pi].imshow(maps[:, :, pos[pi], i], cmap=plt.cm.jet)
                    ax[i, pi].set_title('X pos = %d'%(pos[pi]))
                    ax2[i, 0].plot([pos[pi], pos[pi]], [0, y_points], 'r--', linewidth=1.5)
                    ax2[i, 1].plot([pos[pi], pos[pi]], [0, z_points], 'r--', linewidth=1.5)
                    
        if 'g_cscale' in kwargs.keys():
            scale = kwargs['g_cscale']
            scale = scale if type(scale) is list else [maps.min(), maps.max()]
            for im in imh:
                im.set_clim(scale)
            f.canvas.draw()
            
            
                    
    def get_analysis_data(self):
        maps = self.h5f[self.conv_name][self.analysis_name]['Maps'][:, :, :, :]
        spectra = self.h5f[self.conv_name][self.analysis_name]['Endmembers'][:, :]
        
        return maps, spectra
        
    def calc_topo(self, plot=1, mass=None):
        if 'D' in self.h5f['Raw_data'].keys() and 'Max_peak_position' in self.h5f['Raw_data']['D'].attrs:
            tof_resolution = self.h5f['Raw_data']['D'].attrs['tof_resolution']        
            
            D = self.h5f['Raw_data']['D'][...]
            mp = self.h5f['Raw_data']['D'].attrs['Max_peak_position']        
            if not mass:
                mass = self.h5f['Raw_data']['D'].attrs['Peak_mass']
            
            m_u = 1.66e-27
            e = 1.6e-19
            
            #Acceleration time in ns
            tac = (2*self.h_ex**2*mass*m_u/self.U_ex/e)**0.5 * 1e9
            #Delta ToF in ns
            dT = (mp-mp/D)*tof_resolution*0.05
            
            height = self.h_ex*(1-((tac-dT)/tac)**2)
            if 'Height' in self.h5f['Raw_data'].keys():
                self.h5f['Raw_data']['Height'][...] = height
            else:
                self.h5f['Raw_data'].create_dataset('Height', data=height)
            
            self.h5f['Raw_data']['Height'].attrs['h_ex'] = self.h_ex
            self.h5f['Raw_data']['Height'].attrs['U_ex'] = self.U_ex
            self.h5f['Raw_data']['Height'].attrs['tac'] = tac
            self.h5f['Raw_data']['Height'].attrs['mass'] = mass
            
            if plot:
                f, ax = plt.subplots()
                s = ax.imshow(height[0, :, :]*1e6)
                f.colorbar(s)
                
            self.h5f.flush()
            
            return height
        else:
            print('Shift correction information has not been found in current dataset')
            
    def calc_peak_maps(self, peaks_data, **kwargs):
        """
        peaks_data - list of peaks to be plotted. Each element in the list is tuple
        ('Label', (start_mass, end_mass))
        """
        data = np.zeros((self.z_points, self.x_points, self.y_points, len(peaks_data)))
        labels = []
        start_arr = []
        end_arr = []
        start_mass = []
        end_mass = []       
   
        Data_full = self.h5f[self.conv_name]['Data_full']
        
        for peak, i in zip(peaks_data, range(len(peaks_data))):
            labels += [peak[0]]
            start_ii = np.searchsorted(self.spectra_mass, peak[1][0])
            end_ii = np.searchsorted(self.spectra_mass, peak[1][1])
            t = Data_full[:, start_ii:end_ii].sum(axis=1)
            data[:, :, :, i] = t.reshape((self.z_points, self.x_points, self.y_points))
            start_arr += [start_ii]
            end_arr += [end_ii]
            start_mass += [peak[1][0]]
            end_mass += [peak[1][1]]
            print(i)
        
        m = 1
        for name in self.h5f[self.conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'PeaksArea' and int(parts[1]) >= m:
                m = int(parts[1])+1
        
            
        grp_name = 'PeaksArea_'+str(m).zfill(3)   
        grp = self.h5f[self.conv_name].create_group(grp_name)
        
        grp.create_dataset('Maps', data=data)
        
        asciiList = [label.encode("ascii", "ignore") for label in labels]
        grp.create_dataset('Labels', (len(asciiList), 1), 'S10', asciiList)
        
        grp.create_dataset('start_ii', data=start_arr)
        grp.create_dataset('end_ii', data=end_arr)       
        grp.create_dataset('start_mass', data=start_mass)
        grp.create_dataset('end_mass', data=end_mass)
        
        self.h5f.flush()
        self.analysis_name = grp_name
        
        if 'plot' in kwargs.keys() and kwargs['plot']:
            self.plot_loading_maps()   
            
    def _data_preparation(self, **kwargs):
        if 'save_mode' in self.h5f[self.conv_name].attrs:
            smode = self.h5f[self.conv_name].attrs['save_mode']
            if smode == 'all' or smode == b'all':
                z_bins = self.h5f[self.conv_name].attrs['z_bins']
                lateral_points = self.x_points*self.y_points
                data = np.zeros((self.z_points*lateral_points, self.spectra_len))
                for i in range(self.z_points):
                    for j in range(z_bins):
                        data[i*lateral_points:(i+1)*lateral_points, :] += \
                         self.h5f[self.conv_name]['Data_full'][(i*z_bins+j)*lateral_points:(i*z_bins+j+1)*lateral_points, :]
            elif smode == 'bin' or smode == b'bin':       
                data = self.h5f[self.conv_name]['Data_full'][...]
        else: data = self.h5f[self.conv_name]['Data_full'][...]
        
        data4d = data.reshape((self.z_points, self.x_points, self.y_points, -1))
        (z_points, x_points, y_points, s_points) = data4d.shape
                
        if 'spatial_range' in kwargs.keys():
            #Range in format (z_range, x_range, y_range)
            srange = kwargs['spatial_range']
            ii_z = slice(self.z_points) if srange[0] == -1 else srange[0]
            ii_x = slice(self.x_points) if srange[1] == -1 else srange[1]
            ii_y = slice(self.y_points) if srange[2] == -1 else srange[2]
            data4d = data4d[ii_z, ii_x, ii_y]
            (z_points, x_points, y_points, s_points) = data4d.shape
            data = data4d.reshape((-1, s_points))  
                  
        if 'spectral_range' in kwargs.keys():
            srange = kwargs['spectral_range']
            data4d = data4d[:, :, :, srange]
            (z_points, x_points, y_points, s_points) = data4d.shape
            data = data4d.reshape((-1, s_points)) 
            
        if 'norm' in kwargs.keys():
            if kwargs['norm'] == 'depth':
                data2dz = data4d.reshape((z_points, -1))
                data4d = (data2dz.T/np.linalg.norm(data2dz, axis=1)).T.reshape((z_points, x_points, y_points, s_points))
                data = data4d.reshape((-1, s_points))
            elif kwargs['norm'] == 'all':
                data = (data.T/np.linalg.norm(data, axis=1)).T
                data4d = data.reshape((z_points, x_points, y_points, s_points))
                
        return data4d
            
    def PCA(self, comp_num, **kwargs):
        """
        PCA of selected converted data
        comp_num - limit of calculated components
        """
        print('PCA data preparation...')
        data4d = self._data_preparation(**kwargs)
        (z_points, x_points, y_points, s_points) = data4d.shape
        if z_points == 0: z_points = 1
        data = data4d.reshape((-1, s_points))        

            
        print('PCA claculation...')      
        model = PCA(n_components=comp_num)
        maps = model.fit_transform(data).reshape((z_points, x_points, y_points, -1))
        spectra = model.components_
        
        
        #Saving data
        m = 1
        for name in self.h5f[self.conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'PCA' and int(parts[1]) >= m:
                m = int(parts[1])+1
                
        grp_name = 'PCA_'+str(m).zfill(3)        
        grp = self.h5f[self.conv_name].create_group(grp_name)
        grp.attrs['Number_of_components'] = comp_num
        grp.attrs['Mode'] = 'PCA over whitened data with limited number of components'
        
        if 'spatial_range' in kwargs.keys():
            grp.attrs['x_points'] = x_points
            grp.attrs['y_points'] = y_points
            grp.attrs['z_points'] = z_points
            grp.attrs['s_points'] = s_points
        
        
        grp.create_dataset('Maps', data=maps)
        grp.create_dataset('Endmembers', data=spectra)
        
        self.h5f.flush()
        self.analysis_name = grp_name
        
        #Plotting data
        if 'plot' in kwargs.keys() and kwargs['plot']:
            self.plot_endmembers()
            self.plot_loading_maps()

class SIMSmodel(object):
    '''
    Contains data required for SIMS data conversion
    '''
    def __init__(self, tof_resolution=4, xy_bins=1, z_bins=1, z_points=None,
                 width_threshold=4, counts_threshold=1000, z_range=None, save_mode='bin'):
        '''
        Initializes model

        Inputs:
        --------
        tof_resolution : int
            Tof binning resolution (default tof_resoulution=4)
        xy_bins : int
            Lateral binning resolution (default xy_bins=1)
        width_threshold : int
            Minimal width of the detected peak (default width_threshold=4)
        counts_threshold : int
            Minmal height of the detected peak (default counts_threshold=1000)
        z_range : tuple (z_start,z_end)
            Range of processed data
        '''
        self.tof_resolution = tof_resolution
        self.xy_bins = xy_bins
        self.width_threshold = width_threshold
        self.counts_threshold = counts_threshold

        self.z_points = z_points
        self.z_bins = z_bins

        #No shift correction by default
        self.shift_corr = 0

        #All peaks are converted by default
        self.conv_all = 1
        self.exclude_mass = []

        self.z_range = z_range

        self.save_mode = save_mode
        
        self.peaks_mass = []
        self.corr_peak = []

    def enable_shift_correction(self, corr_peak=None):
        '''
        Enables shift correction in the data conversion workflow

        Inputs:
        --------
        corr_peak : float
            Mass corresponding to peak used for ToF correction
        '''
        self.shift_corr = 1
        self.corr_peak = corr_peak

    def disable_shift_correction(self):
        '''
        Disables shift correction in the data conversion workflow
        '''
        self.shift_corr = 0

    def convert_all_peaks(self, exclude_mass=None):
        '''
        Selects all peaks, excluding specified in exclude_mass to be converted

        Input:
        --------
        exclude_mass : list
            Masses to be excluded from conversion process
        '''
        self.conv_all = 1
        self.exclude_mass = exclude_mass

    def convert_few_peaks(self, peaks_mass):
        '''
        Selects few peaks with masses specified in peaks_mass

        Input:
        --------
        peaks_mass : list
            Masses to be included to conversion process
        '''
        self.conv_all = 0
        self.peaks_mass = peaks_mass
        
class SIMSPeakBrowser(object):
    def __init__(self, fig, ax, tof, mass, data, f_back = None):
        self.fig=fig
        self.ax=ax
        self.excluded_peaks=[]

        self.peaks=[]        
        self.f_back=f_back
        i=0
        while i<tof.size:
            i0=i
            while i<tof.size-1 and tof[i+1]==tof[i]+1:
                i+=1
            self.peaks+=[[range(i0,i+1),mass[i0:i+1],data[i0:i+1],False]]
            i+=1
        self.plot()
            
        
    def plot(self):
        m0=0
        i0=0
        self.ax[0].cla()
        self.ax[1].cla()
        for i in range(len(self.peaks)):
            self.ax[0].plot([i0,self.peaks[i][0][0]],[0,0],linewidth=1.5,color='b')
            
            self.ax[0].plot(self.peaks[i][0],self.peaks[i][2],linewidth=1.5,
                        color=('r' if self.peaks[i][3] else 'b'))            
            i0=self.peaks[i][0][-1]
                        
            self.ax[1].plot([m0,self.peaks[i][1][0]],[0,0],linewidth=1.5,color='b')
            self.ax[1].plot(self.peaks[i][1],self.peaks[i][2],linewidth=1.5,
                        color=('r' if self.peaks[i][3] else 'b'))            
            m0=self.peaks[i][1][-1]
        plt.draw()        
            
        
    def onclick(self, event):
        if event.dblclick:
            pos=event.xdata
            
            ax_id=0 if event.inaxes==self.ax[0] else 1
            
            i=0
            while i<len(self.peaks) and (pos<=self.peaks[i][ax_id][0] or pos>=self.peaks[i][ax_id][-1]):
                i+=1
            if i<len(self.peaks):
                self.peaks[i][3]=False if self.peaks[i][3] else True
                self.excluded_peaks+=[self.peaks[i][1].mean()]
                self.plot()                
                
        elif event.button==3:
            self.f_back(self.excluded_peaks)
            plt.close(self.fig)
            
class SIMSrawdata(object):
    '''
    Class stroing and handling raw ToF SIMS data
    '''
    def load_from_file(self, path, file_prefix):
        '''
        Loads raw SIMS data from GRD files
        
        Inputs
        --------
        path : String or unicode
            Path to folder with GRD files
        file_prefix : String or unicode
            Prefix of the dataset name
        '''
        scans_path=path+'\\'+file_prefix+'.itm.scans'
        coords_path=path+'\\'+file_prefix+'.itm.coords'
        tofs_path=path+'\\'+file_prefix+'.itm.tofs'
            
        f=open(scans_path, 'rb')
        self.z=np.fromstring(f.read(),dtype='uint32')
        f.close()
        
        f=open(coords_path, 'rb')
        coords=np.fromstring(f.read(),dtype='uint32')
        f.close()
        self.x=coords[::2]
        self.y=coords[1::2]
        del(coords)
        
        f=open(tofs_path, 'rb')
        self.tofs=np.fromstring(f.read(),dtype='uint32')
        f.close()        
        
        parms_path=path+'\\'+file_prefix+'.properties.txt'
        f=open(parms_path)
        self.sims_parms={}    
        
        for line in f:
            words=line.split()
            self.sims_parms[words[0]]=float(words[2])               
        f.close()
    
        self.SF=self.sims_parms['Context.MassScale.SF']
        self.K0=self.sims_parms['Context.MassScale.K0']
        self.scan_size=self.sims_parms['Registration.Raster.FieldOfView']*1e6
        
        self.layers=self.z.max()+1
        self.cols=self.x.max()+1
        self.rows=self.y.max()+1
        
        self.counts=self.tofs.size
        
    def load_from_array(self, x,y,z,tofs, SF, K0, scan_size):
        '''
        Loads raw SIMS data numpy arrays
        
        Inputs
        --------
        x,y,z,tofs : 1D numpy arrays
            x,y,z coordinates and corresponidng times of flight
        SF, K0 : float
            SF and K0 callibration coefficients
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
        
        if 'Raw_data' in h5f.keys():
            del h5f['Raw_data']

        grp=h5f.create_group('Raw_data')
        
        dset=grp.create_dataset('Raw_data', (self.counts, 4), dtype='uint32',
                                chunks=(self.counts, 1))    #Implement chunking of 1-D data
        dset[:,0]=self.x
        dset[:,1]=self.y
        dset[:,2]=self.z
        dset[:,3]=self.tofs
        
        for name,val in self.sims_parms.items():
            dset.attrs[name]=val
            
        grp.attrs['Total_counts']=self.counts
        grp.attrs['SF']=self.SF
        grp.attrs['K0']=self.K0
        grp.attrs['x_points']=self.cols
        grp.attrs['y_points']=self.rows
        grp.attrs['z_points']=self.layers
            
        h5f.flush()
        
class SIMStranslator(Translator):
    '''
    Legacy Data Format translator
    '''
    
    def translate(self, folder_path, file_prefix, conv_model, cores=2):
        '''
        Translates the provided SIMS dataset to a .h5 file
        
        Inputs
        --------
        path : String or unicode
            Path to folder with GRD files
        file_prefix : String or unicode
            Prefix of the dataset name
        conv_model : SIMSmodel
            Model used for data conversion
        '''
        
        
        raw_data=SIMSrawdata()
        raw_data.load_from_file(folder_path, file_prefix)               

        self.sims_data=SIMSdata(raw_data, conv_model)        
        self.sims_data.data_conversion(cores=cores)
        
        # example of important values you might extract from the parameters:
        size_x = raw_data.scan_size # this would be in microns for example
        size_y = raw_data.scan_size # this would be in microns for example
        num_x_pts = self.sims_data.x_points
        num_y_pts = self.sims_data.y_points
        num_spec_inds = self.sims_data.spectra_len
               
        # Make position Index matrix:
        (pos_ind_mat, pos_labs) = makePositionMat([num_x_pts, num_y_pts],['X','Y'])
        # Introduce dimension labels - we have an automated way of doing this in the new version
        pos_slices = getPositionSlicing(pos_labs) 
        
        # Make position values matrix:
        pos_val_mat = np.float32(pos_ind_mat)
        pos_val_mat[0,:] = pos_val_mat[0,:]*(size_x/num_x_pts)
        pos_val_mat[1,:] = pos_val_mat[1,:]*(size_y/num_y_pts)
        
        # Make Spectroscopic Index matrix:
        # in the case of SIMS we only have one spectroscopic dimension but we expect a 2D matrix
        spec_ind_mat = np.atleast_2d(np.arange(num_spec_inds, dtype=np.uint32))
        spec_slices = {'Mass':(slice(None),slice(0,1))}
        # Make the last of the 4 ancillary datasets:
        spec_val_mat = np.atleast_2d(self.sims_data.spectra_mass)
        
        # Now we prepare the tree structure of the h5 file
        # we build the tree from bottom to top so first prepare the leaves
        ''' 
        Currently PySPM allows 3 modes of writing datasets:
            method 1 - (used for ancillary datasets that are typically small)
                        We just assign the data when creating an AFMDataset container
            method 2 - (used for the main dataset in this example) 
                        we know the exact size of the data and we allocate the space
                        We populate the dataset after writing the tree (skeleton)
            method 3 - (not shown here but see BEPSndfTranslator)
                        Sizes of the raw data are unknown
        '''
        
        # first the ancillary datasets
        ds_tofs_calibr = AFMDataset('ToF callibration', self.sims_data.spectra_tofs)     
        ds_mass_calibr = AFMDataset('Mass callibration', self.sims_data.spectra_mass)
                
        # these datasets are mandatory and MUST have these exact names
        ds_pos_inds = AFMDataset('Position_Indices', pos_ind_mat)
        ds_pos_inds.attrs['labels'] = pos_slices
        ds_pos_vals = AFMDataset('Position_Values', pos_val_mat)
        ds_pos_vals.attrs['labels'] = pos_slices           
        ds_spec_inds = AFMDataset('Spectroscopic_Indices', spec_ind_mat)
        ds_spec_inds.attrs['labels'] = spec_slices          
        ds_spec_vals = AFMDataset('Spectroscopic_Values', spec_val_mat)            
        ds_spec_vals.attrs['labels'] = spec_slices

        ds_main_data = AFMDataset('Main_Data', chunking=(num_spec_inds,1), dtype=np.float32,
                                data=self.sims_data.data2d.T.astype('float32'))
        ds_rawdata = AFMDataset('SIMS_raw_data', dtype=np.uint32, 
                                data=raw_data.get_2d_rawdata().astype(np.uint32))
        
        # Now assemble the tree structure:
        meas_grp = AFMDataGroup('Measurement_000')
        meas_grp.addChildren([ds_main_data,ds_spec_inds,ds_spec_vals,ds_pos_inds,
                              ds_pos_vals,ds_tofs_calibr,ds_mass_calibr,ds_rawdata])
                              
        meas_grp.attrs = raw_data.sims_parms
        
        spm_data = AFMDataGroup('')
        global_parms = generateDummyMainParms() # dummy parameters the describe the file itself
        global_parms['data_type'] = 'SIMS'
        global_parms['translator'] = 'SIMS'
        
        global_parms['grid_size_x'] = num_x_pts
        global_parms['grid_size_y'] = num_y_pts

        spm_data.attrs = global_parms
        spm_data.addChildren([meas_grp])
        
        print('Writing following tree:')
        meas_grp.showTree()
        
        # decide the file name and folder path:
        #(waste, basename) = path.split(folder_path)
        h5_path = path.join(folder_path,file_prefix + '.h5')
        
        # Remove any previously translated file:
        if path.exists(h5_path):
            remove(h5_path)
            
        # Instantiate the fancy HDF5 writer to write this tree
        hdf = ioHDF5(h5_path)
        h5_refs = hdf.writeData(spm_data)
        
        # Now link the mandatory ancillary datasets to the main dataset:
        aux_ds_names = ['Position_Indices','Position_Values',
                        'Spectroscopic_Indices','Spectroscopic_Values']
        # We know there is exactly one main data
        h5_main = getH5DsetRefs(['Main_Data'], h5_refs)[0]             
        # Reference linking now         
        hdf.linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))

        # close the file and return
        hdf.flush()
        hdf.close()
        
        return h5_path