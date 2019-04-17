# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:44:23 2017

@author: iv1
"""

from SIMSdata import SIMSdata

#Path to h5 file
h5_path=r"E:\TylerData\SIMSdata\testdata\002 stage scan.h5"

#Loading data
sims=SIMSdata()        
sims.load_h5(h5_path)

#Listed datasets in this h5
sims.list_converted_data()

#Loading first one and plotting averaged data
sims.load_converted_data(1)
#sims.plot_ave_data()


#PCA on the data
sims.PCA(comp_num=5, spatial_range=(slice(30),-1,-1))
#sims.PCA(comp_num=5, spatial_range=(slice(1,2),-1,-1))


#Plotting PCA results

#sims.plot_endmembers(mode='split')

#sims.plot_endmembers(mode='merge', delta=0.25)

#sims.plot_loading_maps(mode='cross',pos=(64,64,0))
#sims.plot_loading_maps(mode='cross',pos=(-1,-1,-1))
#sims.plot_loading_maps(mode='tomo',direction='XY',pos=range(0,25,5))
#sims.plot_loading_maps(mode='tomo',direction='XY',pos=list(range(0,2,1)))
#sims.plot_loading_maps(mode='tomo',direction='YZ',pos=range(0,25,5))

sims.close()