#Zhang MQ, Mar 2021

import os
import math
import pickle

import numpy  as np
import pandas as pd
import xarray as xr
import shelve as sh

import matplotlib
matplotlib.use('Agg')  # turn off MPL backend to avoid x11 requirement
import matplotlib.pyplot    as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape
    
    fCpr = xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.pre.dat.nc')
    Latc, Lonc = fCpr.lat, fCpr.lon

    with open ('China_historical_tmax_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his = data_dict['bigMax_ann_historical']

    with open ('China_historical_tmin_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ann_his = data_dict['bigMin_ann_historical']

    with open('../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = data_dict['bigMax_all'], data_dict['bigMin_all']

    c_max = 0
    c_min = 0
    c_china = 0
    
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                c_china = c_china + 1
                
                fpr = xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.pre.dat.nc')
                cur_lat, cur_lon = lat[j], lon[i]
                c_lat, c_lon = fpr.lat.sel(lat=cur_lat, method='nearest'), fpr.lon.sel(lon=cur_lon, method='nearest')
                cc_lat, cc_lon = np.where(fpr.lat == c_lat), np.where(fpr.lon == c_lon)
                latc, lonc = int(cc_lat[0]), int(cc_lon[0])

                std_cru_max = np.std(bigMax_cru_his[:, latc, lonc].values.reshape(-1))
                std_cru_min = np.std(bigMin_cru_his[:, latc, lonc].values.reshape(-1))
                std_ann_max = np.std(bigMax_ann_his[:, :, j, i].reshape(-1))
                std_ann_min = np.std(bigMin_ann_his[:, :, j, i].reshape(-1))
                print(std_cru_max)
                print(std_ann_max)
                print(std_cru_min)
                print(std_ann_min)
                print('1')

                if std_ann_max < std_cru_max:
                    c_max = c_max + 1
                if std_ann_min < std_cru_min:
                    c_min = c_min + 1
                else:
                    continue

    print(c_china)
    print(c_max)
    print(c_min)








