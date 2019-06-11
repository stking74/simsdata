# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:52:44 2016

@author: iv1
"""

import numpy as np
import h5py
from numpy import pi
from scipy.optimize import curve_fit
from multiprocessing import Lock

def maxwell(x, *p):
    A, a, x0 = p
    res=A*np.sqrt(2/pi)*(x-x0)**2*np.exp(-(x-x0)**2/(2*a**2))/a**3
    res[np.where(x<x0)]=0.0
    return res
    
def r_square(xdata, ydata, func, popt):
    y_mean=ydata.mean()
    SS_tot=np.sum((ydata-y_mean)**2)
    SS_res=np.sum((ydata-func(xdata,*popt))**2)
    return 1-SS_res/SS_tot

def convert_chunk(p):
    """
    p[0] - chunk start position
    p[1] - (h5 file path,chunk size,xy_bins, z_bins, signal_ii, tof_resolution)
    """
    h5f=h5py.File(p[1][0],'r')
    counts=h5f['Raw_data']['Raw_data'].shape[0]
    
    #lock = Lock()
    
    try:
        lock.acquire()
        data_chunk=h5f['Raw_data']['Raw_data'][p[0]:(p[0]+p[1][1]),:] if (p[0]+p[1][1])<counts else\
            h5f['Raw_data']['Raw_data'][p[0]:,:]
        lock.release()
    except NameError: 
        data_chunk=h5f['Raw_data']['Raw_data'][p[0]:(p[0]+p[1][1]),:] if (p[0]+p[1][1])<counts else\
            h5f['Raw_data']['Raw_data'][p[0]:,:]
            
    h5f.close()
    xy_bins, z_bins, signal_ii, tof_resolution=p[1][2:]  
    data_out=[]
    
    i=0
    while i<data_chunk.shape[0]:
        x,y,z=data_chunk[i,:-1]
        point_spectrum=np.zeros(len(signal_ii)+3).astype('uint32')
        point_spectrum[:3]=(int(x/xy_bins),int(y/xy_bins),int(z/z_bins))
        while i<data_chunk.shape[0] and tuple(data_chunk[i,:-1])==(x,y,z):
            c_tof=int(data_chunk[i,3]/tof_resolution)              
            ii=np.searchsorted(signal_ii, c_tof) 
            if (ii<len(signal_ii)) and (signal_ii[ii]==c_tof):
                point_spectrum[ii+3]+=1
            i+=1
            
        data_out+=[point_spectrum]    
    return data_out


def correct_shift(p):
    x=p[0]
    y=p[1]
    z=p[2]
    tof=p[3]
    
    D=p[4][0]    
    xy_bins=p[4][1]
    z_bins=p[4][2]    
  
    tof_corr=np.copy(tof)
    print(D.shape)
    print(len(tof))
    
    for i in range(len(tof)):
        
        d=D[int(z[i]/z_bins),int(x[i]/xy_bins),int(y[i]/xy_bins)]
        tof_corr[i]*=d if not np.isnan(d) else 1
    
    return tof_corr


def fit_point(p):
    ydata=p[0]
    func,xdata=p[1]
    
    p0=[np.max(ydata),1, xdata[np.argmax(ydata)]]    
    
    try:    
        popt, var_matrix=curve_fit(func, xdata, ydata,p0=p0)
        return np.append(popt,r_square(xdata, ydata, func, popt))
    except:
        return(np.zeros(4))    

def init_mp_lock(_lock):
    global lock
    lock=_lock
 
def chunk_spectrum(p):
    '''
    p[0]: start position of the chunk
    p[1]: (h5f_path, chunk_size, tof_resolution, tofs_max)
    '''  
    h5f=h5py.File(p[1][0],'r')
    counts=h5f['Raw_data']['Raw_data'].shape[0]
    
    start_i=p[0]
    chunk_size=p[1][1]
    
    lock.acquire()
    tofs=h5f['Raw_data']['Raw_data'][start_i:(start_i+chunk_size),3] if (start_i+chunk_size)<counts else\
             h5f['Raw_data']['Raw_data'][start_i:,3]
    lock.release()
    
    h5f.close()
    tof_resolution=p[1][2]
    tofs_max=p[1][3]

    spectrum, b=np.histogram(tofs, int(tofs_max/tof_resolution), (0,int(tofs_max/tof_resolution)*tof_resolution))
    return spectrum
    
def peaks_filtering(signal_ii, mass, threshold, exclude_mass):    
    i=0
    filtered_signal_ii=np.empty(0)
    while i<len(signal_ii):
        i0=i
        while i<len(signal_ii)-1 and signal_ii[i+1]==signal_ii[i]+1:
            i+=1
        
        if i-i0+1>=threshold: 
            peak=np.arange(signal_ii[i0], signal_ii[i]+1)    
            for em in exclude_mass:
                if mass[i0]<=em<=mass[i]: peak=np.empty(0)       
            filtered_signal_ii=np.append(filtered_signal_ii, peak)

        i+=1              
    return filtered_signal_ii
    
def peak_labels(signal_ii, ave_spectra, mass):
    i=0
    labels=np.empty(0)
    xpos=np.empty(0)
    ypos=np.empty(0)
    while i<len(signal_ii):
        i0=i
        while i<len(signal_ii)-1 and signal_ii[i+1]==signal_ii[i]+1:
            i+=1
        
        mi=ave_spectra[i0:i].argmax()+i0
        labels=np.append(labels, mass[mi])
        xpos=np.append(xpos,mi)
        ypos=np.append(ypos,ave_spectra[mi])
        i+=1
        
    labels=labels[np.argsort(ypos)[::-1]]
    xpos=xpos[np.argsort(ypos)[::-1]]
    ypos=ypos[np.argsort(ypos)[::-1]]
        
    return labels, xpos, ypos
    
def find_peak(signal_ii, mass, target_mass):
    i=0
    while i<len(signal_ii):
        i0=i
        while i<len(signal_ii)-1 and signal_ii[i+1]==signal_ii[i]+1:
            i+=1
        if mass[i0]<=target_mass and mass[i]>=target_mass:
            return range(i0,i+1)
        i+=1
    
    return None
 

def peaks_split_sort(signal_ii, ave_spectra, mass):   
    i0_arr=np.empty(0,dtype='int32')
    iend_arr=np.empty(0,dtype='int32')
    max_count=np.empty(0)
    max_mass=np.empty(0)
    
    i=0
    while i<len(signal_ii):
        i0=i
        while i<len(signal_ii)-1 and signal_ii[i+1]==signal_ii[i]+1:
            i+=1        
        
        i0_arr=np.append(i0_arr,i0)
        iend_arr=np.append(iend_arr,i)
        
        max_count=np.append(max_count,ave_spectra[i0:i+1].max())
        max_mass=np.append(max_mass,mass[ave_spectra[i0:i].argmax()+i0])

        i+=1
    ii_sort=np.argsort(max_count)[::-1]
    
    i0_arr=i0_arr[ii_sort]
    iend_arr=iend_arr[ii_sort]
    
    ii2d=[range(i0,i+1) for (i0,i) in zip(i0_arr,iend_arr)]   
    max_mass=max_mass[ii_sort]
            
    return ii2d, max_mass  

