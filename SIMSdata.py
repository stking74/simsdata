
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:54:53 2016

@author: iv1
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat, product
from sklearn.decomposition import NMF, PCA
from scipy.ndimage import gaussian_filter
import os

import h5py

from multiprocessing import Pool
from .SIMSconversion import SIMSconversion
from .SIMSrawdata import SIMSrawdata
from .SIMSutils import fit_point, correct_shift

class SIMSdata(object):
    '''
    Class stroing and processing converted SIMS data
    '''

    def __init__(self, h_ex=1.5e-3, U_ex=2e3,cores=2):
        '''
        Initialize SIMSdata data handling class. Begin by setting 
        [[UNKNOWN]] calibration constants.
        
        #!!! Find out what these calibration constants are
        '''
        self.h_ex = h_ex
        self.U_ex = U_ex
        self.cores=cores
        return

    def close(self):
        self.h5f.flush()
        self.h5f.close()

    def load_raw(self, path, file_prefix, nuke=False):
        '''
        Read in SIMS measurement data from raw datafiles.
        
        Input:
        --------
        path : str
            directory containing datafiles
        file_prefix : str
            filename prefix common to all datafiles to be loaded
            
        #!!! Consider implementing a batch mode here?
        '''
        
        #Initialize raw data handler class
        raw_data = SIMSrawdata()
        self.h5f = raw_data.load_from_file(path, file_prefix, nuke=nuke)
        self.h5_path = os.path.join(path, file_prefix+'.h5')
        return

    def load_h5(self, h5_path):
        self.h5_path=h5_path
        self.h5f = h5py.File(self.h5_path)

    def convert_data(self, name, conv_model, cores=5, **kwargs):
        '''
        Starts data conversion process

        Input:
        --------
        name : str
            user-defined name for dataset
        conv_model : SIMSmodel
            initialized data processing model
        cores : int
            number of cores used in conversion process (default cores=5)
        '''
        #Shift correction if enabled
        if conv_model.shift_corr:
            self.shift_correction(conv_model, cores=cores)

        #Convert raw data into Numpy array
        conv_data = SIMSconversion(name, self.h5f, conv_model, self.h5_path,cores=self.cores)

        if conv_model.conv_all:
            conv_data.select_all_peaks(conv_model.exclude_mass)
        else:
            conv_data.select_few_peaks(conv_model.peaks_mass)

        if cores < 2:
            print('Single_process')
            conv_data.convert_single_process()
        else:
            print('Multi_process')
            conv_data.convert_multiprocessing(cores=cores)

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
        conv_model : SIMSmodel
            initialized data processing model
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
        return
    
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
            f, ax = plt.subplots(comp_num, 1, sharex=True, sharey=True)
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
        if not hasattr(self,'analysis_name'):
            self.analysis_name = 'PCA_'+str(1).zfill(3)
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
            if s_points == 0: data = data4d.reshape((-1))
            else: data = data4d.reshape((-1, s_points))

        if 'spectral_range' in kwargs.keys():
            srange = kwargs['spectral_range']
            data4d = data4d[:, :, :, srange]
            (z_points, x_points, y_points, s_points) = data4d.shape
            if s_points == 0: data = data4d.reshape((-1))
            else: data = data4d.reshape((-1, s_points))

        if 'norm' in kwargs.keys():
            if kwargs['norm'] == 'depth':
                data2dz = data4d.reshape((z_points, -1))
                data4d = (data2dz.T/np.linalg.norm(data2dz, axis=1)).T.reshape((z_points, x_points, y_points, s_points))
                if s_points == 0: data = data4d.reshape((-1))
                else: data = data4d.reshape((-1, s_points))
            elif kwargs['norm'] == 'all':
                data = (data.T/np.linalg.norm(data, axis=1)).T
                data4d = data.reshape((z_points, x_points, y_points, s_points))
                
        return data4d
            
    def PCA(self, comp_num, **kwargs):
        """
        PCA of selected converted data
        comp_num - limit of calculated components
        """
        m = 1
        grp_name = 'PCA_'+str(m).zfill(3)   
        if grp_name in self.h5f.keys():
            print('PCA has already been performed for this dataset.')
            answer = raw_input('Repeat PCA? y/[n]')
            if answer == '' or answer.upper() == 'N':
                print('PCA aborted')
                return
            elif answer.upper() == "Y":
                while grp_name in self.h5f.keys():
                    m+=1
                    grp_name = 'PCA_'+str(m).zfill(3)  
            else:
                print('Answer not recognized, PCA aborted.')
                return
        print('PCA data preparation...')
        data4d = self._data_preparation(**kwargs)
        (z_points, x_points, y_points, s_points) = data4d.shape
        if z_points == 0: z_points = 1
        if s_points == 0: data = data4d.reshape((-1))
        else: data = data4d.reshape((-1, s_points)) 
        
        print('PCA claculation...')      
        model = PCA(n_components=comp_num)
        maps = model.fit_transform(data).reshape((z_points, x_points, y_points, -1))
        spectra = model.components_
        
        #Saving data
        for name in self.h5f[self.conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'PCA' and int(parts[1]) >= m:
                m = int(parts[1])+1
                    
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

    def IncrementalPCA(self, comp_num, **kwargs):
        m = 1
        grp_name = 'PCA_'+str(m).zfill(3)   
        if grp_name in self.h5f.keys():
            print('PCA has already been performed for this dataset.')
            answer = raw_input('Repeat PCA? y/[n]')
            if answer == '' or answer.upper() == 'N':
                print('PCA aborted')
                return
            elif answer.upper() == "Y":
                while grp_name in self.h5f.keys():
                    m+=1
                    grp_name = 'PCA_'+str(m).zfill(3)  
            else:
                print('Answer not recognized, PCA aborted.')
                return
        print('PCA data preparation...')
        data4d = self._data_preparation(**kwargs)
        (z_points, x_points, y_points, s_points) = data4d.shape
        if z_points == 0: z_points = 1
        data = data4d.reshape((-1, s_points))
        data_height = s_points
        
        print('Training PCA...')  
        batch_size = 1000
        n_batches = data_height // batch_size
        model = PCA(n_components=comp_num, batch_size=batch_size)
        for i in n_batches:
            low = i * batch_size
            high = low + batch_size
            batch = data[...,low:high]
            model.partial_fit(batch)
        batch = data[...,high:]
        model.partial_fit(batch)
        
        
        print('Mapping spectra to PCA...')
        data4d.reshape((z_points, x_points, y_points, -1))
        spectra = model.components_
        
        #Saving data
        for name in self.h5f[self.conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'PCA' and int(parts[1]) >= m:
                m = int(parts[1])+1
                    
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
