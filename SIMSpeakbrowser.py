# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:31:22 2016

@author: iv1
"""

import matplotlib.pyplot as plt

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