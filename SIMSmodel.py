# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:03:47 2016

@author: iv1
"""

class SIMSmodel(object):
    '''
    Contains data required for SIMS data conversion
    '''
    def __init__(self, tof_resolution=4, xy_bins=1, z_bins=1, z_points=None,
                 width_threshold=4, counts_threshold=1000, z_range=None, save_mode='bin', 
                 cores=1, chunk_size=1e6):
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
        self.cores=cores
        self.chunk_size=int(chunk_size)
        return

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
        