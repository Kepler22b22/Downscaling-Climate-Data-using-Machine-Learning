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


def weight_mean_ground(big_ann):
    ny_, nd_, clat, clon = big_ann.shape
    big_ann_new = np.zeros((ny_, nd_, Nlat, Nlon))
    total_w = 0.0
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                clat = math.radians(lat[j])
                w = np.cos(clat)
                big_ann_new[:, :, j, i] = big_ann[:, :, j, i] * w
            else:
                big_ann_new[:, :, j, i] = np.nan
    for a in range(Nlat):
        for b in range(Nlon):
            if m[a, b]:
                lat_ = math.radians(lat[a])
                total = np.cos(lat_)
                total_w = total_w + total
    big_ann_new = big_ann_new.reshape(ny_, nd_, 160*280)
    big_ann_new = np.nansum(big_ann_new, axis=2)
    big_ann_new = big_ann_new / total_w
    return big_ann_new


def weight_mean(big_ann_historical):
    big_ann_new = np.zeros((56, 365, Nlat, Nlon))
    total_w = 0.0
    for j in range(Nlat):
        clat = math.radians(lat[j])
        w = np.cos(clat)
        big_ann_new[:, :, j, :] = big_ann_historical[:, :, j, :] * w
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                lat_ = math.radians(lat[j])
                total = np.cos(lat_)
                total_w = total_w + total
    big_ann_new = big_ann_new.reshape(56, 365, 160*280)
    big_ann_new = np.nansum(big_ann_new, axis = 2)
    big_ann_new = big_ann_new / total_w
    return big_ann_new


def weight_mean_ccsm(big_ccsm_historical):
    ny_, nd_, clat, clon = big_ccsm_historical.shape
    big_ccsm_new = np.zeros((ny_, nd_, clat, clon), dtype='float')
    total_w = 0.0
    fCpr = xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_tasmax_1948-2005.nc')
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
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
                big_ccsm_new[:, :, j, i] = big_ccsm_historical[:, :, j, i] * w
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
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    with open('./China_historical_tmax_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_raw = data_dict['bigMax_ann_historical']
    
    with open('./China_historical_tmin_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ann_raw = data_dict['bigMin_ann_historical']

    with open('../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ground_tr, bigMax_ccsm_tr = \
        data_dict['bigMax_ground_tr'], data_dict['bigMax_ccsm_tr']
    
    with open('../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ground_tr, bigMin_ccsm_tr = \
        data_dict['bigMin_ground_tr'], data_dict['bigMin_ccsm_tr']

    bigMax_ann_raw, bigMin_ann_raw = \
      bigMax_ann_raw.reshape(56, 365, Nlat, Nlon), \
      bigMin_ann_raw.reshape(56, 365, Nlat, Nlon)

    bigMax_ground_tr, bigMin_ground_tr = \
      bigMax_ground_tr.values.reshape(56, 365, Nlat, Nlon), \
      bigMin_ground_tr.values.reshape(56, 365, Nlat, Nlon)

    bigMax_ccsm_tr, bigMin_ccsm_tr = \
      bigMax_ccsm_tr.values.reshape(56, 365, 42, 57), \
      bigMin_ccsm_tr.values.reshape(56, 365, 42, 57)

    bigMax_ann_raw, bigMin_ann_raw = weight_mean(bigMax_ann_raw), weight_mean(bigMin_ann_raw)
    bigMax_ground_tr, bigMin_ground_tr = weight_mean_ground(bigMax_ground_tr), weight_mean_ground(bigMin_ground_tr)
    bigMax_ccsm_tr, bigMin_ccsm_tr = weight_mean_ccsm(bigMax_ccsm_tr), weight_mean_ccsm(bigMin_ccsm_tr)

    with open('China_his_temp_weightmean_else.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_raw': bigMax_ann_raw, 
                     'bigMin_ann_raw': bigMin_ann_raw, 
                     'bigMax_ground_tr': bigMax_ground_tr, 
                     'bigMin_ground_tr': bigMin_ground_tr, 
                     'bigMax_ccsm_tr': bigMax_ccsm_tr, 
                     'bigMin_ccsm_tr': bigMin_ccsm_tr}, f, protocol = 4)









