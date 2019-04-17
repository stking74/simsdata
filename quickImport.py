# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:02:36 2018

@author: s4k
"""

#%%This code won’t work from Python console

from SIMSmodel import SIMSmodel
from SIMSdata import SIMSdata
import os.path

#%%Path to file and prefix
path = r'E:\TylerData\SIMSdata\testdata'
prefix = '002 stage scan'
h5_path = os.path.join(path,prefix+'.h5')
#%%
sims_data = SIMSdata()
sims_data.load_h5(h5_path)
#%%
sims_data.list_converted_data()
sims_data.load_converted_data(1)
#%%
sims_data.plot_loading_maps(mode='cross',pos=(64,64,0))
#%%
sims_data.plot_loading_maps(mode='cross',pos=(-1,-1,-1))
#%%
sims_data.plot_loading_maps(mode='tomo',direction='XY',pos=range(0,25,5))
#%%
sims_data.plot_loading_maps(mode='tomo',direction='XY',pos=list(range(0,2,1)))
#%%
sims_data.plot_endmembers()
sims_data.plot_loading_maps()
#%%
sims_data.NMF(3)