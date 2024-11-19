# Zhang MQ, Aug 2020


import os
import math
import pickle

import matplotlib.pyplot as plt
import xarray            as xr
import numpy             as np
import skill_metrics     as sm

from matplotlib            import rcParams
from sys                   import version_info
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def monthlymean(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    ny_, nd_ = bigArray.shape
    #ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    #bigArray = bigArray.reshape(ny_, 365, nlat_, nlon_)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


def weight_mean_ccsm(big_ccsm_historical):
    ny_, nd_, clat, clon = big_ccsm_historical.shape
    big_ccsm_new = np.zeros((ny_, nd_, clat, clon), dtype='float')
    total_w = 0.0
    fCpr = xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_pr_1948-2005.nc')
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

    
def weight_mean_bc(big_historical):
    big_new = np.zeros((56, 365, Nlat, Nlon))
    total_w = 0.0
    for j in range(Nlat):
        clat = math.radians(lat[j])
        w = np.cos(clat)
        big_new[:, :, j, :] = big_historical[:, :, j, :] * w
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                lat_ = math.radians(lat[j])
                total = np.cos(lat_)
                total_w = total_w + total
    big_new = big_new.reshape(56, 365, 160*280)
    big_new = np.nansum(big_new, axis = 2)
    big_new = big_new / total_w
    return big_new


def weight_mean_cru(big_cru_his):
    ny_, nd_, clat, clon = big_cru_his.shape
    big_cru_new = np.zeros((ny_, nd_, clat, clon), dtype='float')
    total_w = 0.0
    fCpr = xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.pre.dat.nc')
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    Latc, Lonc = fCpr.lat, fCpr.lon
    for j in range(360):
        for i in range(720):
            cur_lat, cur_lon = Latc[j], Lonc[i]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                clat = math.radians(int(cur_lat))
                w = np.cos(clat)
                big_cru_new[:, :, j, i] = big_cru_his[:, :, j, i] * w
            else:
                big_cru_new[:, :, j, i] = np.nan
    for j in range(360):
        for i in range(720):
            cur_lat, cur_lon = Latc[j], Lonc[i]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                lat_ = math.radians(int(cur_lat))
                total = np.cos(lat_)
                total_w = total_w + total
    big_cru_new = big_cru_new.reshape(ny_, nd_, 360*720)
    big_cru_new = np.nansum(big_cru_new, axis=2)
    big_cru_new = big_cru_new / total_w
    return big_cru_new


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Read data from pickle file
    with open('../../01.bc.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his = data_dict['bigPr_bc_his']

    with open('../../../data/ccsm4/future/tr_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_his = data_dict['bigPr_ccsm_tr']

    with open('../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_cru_his = data_dict['bigPr_all']

    #bigPr_ccsm_his = bigPr_ccsm_his.values.reshape(56, 365, 42, 57)

    #bigPr_bc_his = weight_mean_bc(bigPr_bc_his)
    #bigPr_ccsm_his = weight_mean_ccsm(bigPr_ccsm_his)

    #bigPr_cru_his = bigPr_cru_his.values.reshape(56, 12, 360, 720)
    #bigPr_cru_his = bigPr_cru_his[20:50, :, :, :]
    #bigPr_cru_his = weight_mean_cru(bigPr_cru_his)

    #mm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #for i in range(12):
    #    bigPr_cru_his[:, i] = bigPr_cru_his[:, i] / mm[i]

    #bigPr_bc_his, bigPr_ccsm_his = \
    #    bigPr_bc_his.reshape(-1) * 86400, \
    #    bigPr_ccsm_his.reshape(-1) * 86400

    #with open('China_talyor_ccsm.pkl', 'wb') as f:
    #    pickle.dump({'bigPr_bc_his': bigPr_bc_his, 
    #                 'bigPr_ccsm_his': bigPr_ccsm_his, 
    #                 'bigPr_cru_his': bigPr_cru_his}, f, protocol=4)

    with open('China_talyor_ccsm.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his, bigPr_ccsm_his, bigPr_cru_his = \
        data_dict['bigPr_bc_his'], data_dict['bigPr_ccsm_his'], data_dict['bigPr_cru_his']

    print(bigPr_bc_his.shape)
    print(bigPr_ccsm_his.shape)
    print(bigPr_cru_his.shape)


