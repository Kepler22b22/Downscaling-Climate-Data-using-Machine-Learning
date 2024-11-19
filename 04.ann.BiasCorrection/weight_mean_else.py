# Zhang MQ, Feb 2020

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


def weight_mean_ccsm(big_ccsm):
    ny_, nd_, clat, clon = big_ccsm.shape
    big_ccsm_new = np.zeros((ny_, nd_, clat, clon), dtype='float')
    total_w = 0.0
    fCpr = xr.open_dataset('../../data/ccsm4/future/ccsm_pr_2006-2100_rcp26.nc')
    fGpr = xr.open_dataset('../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    Lat, Lon = fCpr.lat, fCpr.lon
    for j in range(42):
        for i in range(57):
            cur_lat, cur_lon = Lat[j], Lon[i]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                clat = math.radians(int(cur_lat))
                w = np.cos(clat)
                big_ccsm_new[:, :, j, i] = big_ccsm[:, :, j, i] * w
            else:
                big_ccsm_new[:, :, j, i] = np.nan
    for j in range(42):
        for i in range(57):
            cur_lat, cur_lon = Lat[j], Lon[i]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                lat_ = math.radians(int(cur_lat))
                total = np.cos(lat_)
                total_w = total_w + total
    big_ccsm_new = big_ccsm_new.reshape(ny_, nd_, 42*57)
    big_ccsm_new = np.nansum(big_ccsm_new, axis=2)
    big_ccsm_new = big_ccsm_new / total_w
    return big_ccsm_new


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    #with open('../../data/ccsm4/future/ccsm_rcp_tmax.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ccsm_rcp26, bigMax_ccsm_rcp85 = data_dict['bigMax_ccsm_rcp26'], data_dict['bigMax_ccsm_rcp85']

    #with open('../../../Downscaling-2019-11/data/ccsm4/future/ccsm_rcp_tmin.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ccsm_rcp26, bigMin_ccsm_rcp85 = data_dict['bigMin_ccsm_rcp26'], data_dict['bigMin_ccsm_rcp85']

    #bigMax_ccsm_rcp26, bigMin_ccsm_rcp26 = \
    #  bigMax_ccsm_rcp26.values.reshape(95, 365, 42, 57), \
    #  bigMin_ccsm_rcp26.values.reshape(95, 365, 42, 57)

    #bigMax_ccsm_rcp85, bigMin_ccsm_rcp85 = \
    #  bigMax_ccsm_rcp85.values.reshape(95, 365, 42, 57), \
    #  bigMin_ccsm_rcp85.values.reshape(95, 365, 42, 57)

    #bigMax_ccsm_rcp26, bigMin_ccsm_rcp26 = weight_mean_ccsm(bigMax_ccsm_rcp26), weight_mean_ccsm(bigMin_ccsm_rcp26)
    #bigMax_ccsm_rcp85, bigMin_ccsm_rcp85 = weight_mean_ccsm(bigMax_ccsm_rcp85), weight_mean_ccsm(bigMin_ccsm_rcp85)

    #with open('China_temp_weightmean_else.pkl', 'wb') as f:
    #    pickle.dump({'bigMax_ccsm_rcp26': bigMax_ccsm_rcp26, 
    #                 'bigMin_ccsm_rcp26': bigMin_ccsm_rcp26,
    #                 'bigMax_ccsm_rcp85': bigMax_ccsm_rcp85,
    #                 'bigMin_ccsm_rcp85': bigMin_ccsm_rcp85}, f, protocol = 4)

    with open('China_temp_weightmean_else.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_rcp85 = data_dict['bigMin_ccsm_rcp85']
    print(bigMin_ccsm_rcp85[-1, 0:31])





