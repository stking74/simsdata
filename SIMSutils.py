# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:52:44 2016

@author: iv1
"""

import numpy as np
import h5py
from numpy import pi
from scipy.optimize import curve_fit

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
    p[0] - h5 file path
    p[1] - chunk start position
    p[2] - chunk size
    p[3] - (xy_bins, z_bins, signal_ii, tof_resolution)
    
    """
    h5f=h5py.File(p[0])
    data_chunk=h5f['Raw_data']['Raw_data'][p[1]:(p[1]+p[2]),:]
    h5f.close()
    
    xy_bins, z_bins, signal_ii, tof_resolution=p[3]  
    data_out=[]
    
    i=0
    while i<data_chunk.shape[0]:
        x,y,z=data_chunk[i,:-1]
        point_spectrum=np.zeros(len(signal_ii)+3)
        point_spectrum[:3]=(int(x/xy_bins),int(y/xy_bins),int(z/z_bins))
        while tuple(data_chunk[i,:-1])==(x,y,z):
            c_tof=int(data_chunk[i,3]/tof_resolution)
            if c_tof in signal_ii:
                point_spectrum[c_tof+3]+=1
            i+=1
            
        data_out+=[point_spectrum]
        
    return data_out

def convert_data3d(p):
    x=p[0]
    y=p[1]
    tof=p[2]
    
    x_points, y_points, xy_bins, signal_ii, tof_resolution=p[3]
        
    data2d=np.zeros((x_points*y_points, signal_ii.size), dtype='uint32')    
    
    count=0
    
    sii=np.argsort(tof)
    
    i=0
    while i<tof.size:
        c_tof=tof[sii[i]]
        
        if int(c_tof/tof_resolution) in signal_ii:
            tii=np.searchsorted(signal_ii, int(c_tof/tof_resolution))
            while i<tof.size and tof[sii[i]]==c_tof:
                data2d[int(x[sii[i]]/xy_bins)*y_points+int(y[sii[i]]/xy_bins), tii]+=1
                count+=1
                i+=1
        else:
            while i<tof.size and tof[sii[i]]==c_tof:
                i+=1
    
    return (data2d, count)


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
 
def average_data(p):
    h5f=h5py.File(p[0])
    data_chunk=h5f['Raw_data']['Raw_data'][p[1]:(p[1]+p[2]),:]
    h5f.close()
    tof_resolution,counts_threshold,tofs_max=p[3]
    htofs, b=peaks_detection_hist(data_chunk, tof_resolution, counts_threshold, tofs_max)
    return htofs, b   
    
def peaks_detection(tofs, tof_resolution, threshold):
    M=tofs.max()
    
    htofs, b=np.histogram(tofs, int(M/tof_resolution), (0,int(M/tof_resolution)*tof_resolution))
    max_signal = htofs.max()
    threshold = max_signal * threshold
    signal_ii=np.where(htofs>threshold)[0]
    counts=htofs[signal_ii]
    del(htofs)
    del(b)  
    return signal_ii, counts

def peaks_detection_hist(tofs, tof_resolution, threshold, tofs_max):
    M=tofs_max  
    htofs, b=np.histogram(tofs, int(M/tof_resolution), (0,int(M/tof_resolution)*tof_resolution))
    return htofs, b

def peaks_filtering(signal_ii, mass, threshold, exclude_mass):    
    i=0
    filtered_signal_ii=np.empty(0)
    while i<len(signal_ii):
        i0=i
        while i<len(signal_ii)-1 and signal_ii[i+1]==signal_ii[i]+1:
            i+=1
        
        if i-i0+1>threshold: 
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

def convert_data3d_track(hf, p):
    
    raw_data = hf['Raw_data']['Raw_data']
    
    chunk_size = raw_data.chunks[0]
    
    x_idx = 0
    y_idx = 1
    tof_idx = 3
    
    print("SIMS ToF data sorting...")
    x_points, y_points, xy_bins, signal_ii, tof_resolution = p
    
    #f = open('error_log.txt','w')
        
    data2d=np.zeros((x_points*y_points, signal_ii.size), dtype='uint32')    
    
    count=0
    
    i_max = int(raw_data.shape[0])
    n_blocks = int(i_max // chunk_size)
    remainder = int(i_max % chunk_size)
            
    i=0
    
    complete=0
    
    print("SIMS data binning...")
    for i in range(n_blocks):
        low = int(i * chunk_size)
        high = int(low + chunk_size)
        jj = list(range(low, high))
        c_tofs=raw_data[low:high,tof_idx]
        xs = raw_data[low:high, x_idx]
        ys = raw_data[low:high, y_idx]
        
        for idx, c_tof in enumerate(c_tofs):
        
            if int(c_tof/tof_resolution) in signal_ii:
                tii=np.searchsorted(signal_ii, int(c_tof/tof_resolution))
                index = int(xs[idx]/xy_bins)*y_points+int(ys[idx]/xy_bins)
                try:
                    data2d[index, tii]+=1
                except IndexError as e:
                    #error_line = 'Ping1: '+str(index)+' '+str(tii)+' '+str(e)+'\n'
                    #try: f.write(error_line)
                    #except: pass
                    pass
        print('Finished %i of %i blocks; %04f%%'%(i+1, n_blocks, (i+1)/n_blocks))
                        
#    f.close()
    return (data2d, count)     
