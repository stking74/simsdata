# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:14:27 2016

@author: Suhas Somnath
"""

from __future__ import division # int/int = float
import numpy as np # For array operations
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
        
        print 'Writing following tree:'
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

               
