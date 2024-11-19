#!/usr/bin/env python3
#-------------------------------
# Plot 2D Data on Map
# MQ Zhang, 5 2021

import matplotlib
matplotlib.use('Agg')   # turn off MPL backend to avoid x11 requirement

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import plotfunc               as my
import numpy                  as np
import xarray                 as xr
import matplotlib.pyplot      as plt
import mpl_toolkits.basemap   as bm

from mpl_toolkits.basemap import Basemap


if __name__=='__main__':

    with open('./map_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    yAtr_max, yAtr_min, yCtr_max, yCtr_min, yGtr_max, yGtr_min = \
        data_dict['bigMax_ann'], data_dict['bigMin_ann'], \
        data_dict['bigMax_ccsm'], data_dict['bigMin_ccsm'], \
        data_dict['bigMax_cru'], data_dict['bigMin_cru']

    with open('../../02.bc.plot/map/map_pr.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    yAtr, yCtr, yGtr = \
        data_dict['bigPr_bc'], data_dict['bigPr_ccsm'], data_dict['bigPr_cru']

    #mask
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    fCpr =  xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_pr_1948-2005.nc')
    Lat, Lon = fCpr.lat, fCpr.lon

    fpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    Latc, Lonc = fpr.lat, fpr.lon

    #Tmin
    my.plot2d_map(yGtr_min, Lonc, Latc,
                  levs = np.arange(-40, 20, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-cru-2.pdf' )

    my.plot2d_map(yCtr_min - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic',cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ccsm-2.pdf' )

    my.plot2d_map(yAtr_min - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ann-2.pdf' )

    #Tmax
    my.plot2d_map(yGtr_max, Lonc, Latc,
                  levs = np.arange(-5, 45, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-cru-2.pdf' )

    my.plot2d_map(yCtr_max - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ccsm-2.pdf' )

    my.plot2d_map(yAtr_max - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ann-2.pdf' )
    #Pr
    my.plot2d_map(yGtr, Lonc, Latc,
                  levs = np.arange(0, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-cru-2.pdf' )

    my.plot2d_map(yCtr, Lon, Lat,
                  levs = np.arange(-14, 14, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-ccsm-2.pdf' )

    my.plot2d_map(yAtr, lon, lat,
                  levs = np.arange(-14, 14, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-bc-2.pdf' )




