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


def weight_mean_cru(big_cru_his):
    ny_, nd_, clat, clon = big_cru_his.shape
    big_cru_new = np.zeros((ny_, nd_, clat, clon), dtype='float')
    total_w = 0.0
    fCpr = xr.open_dataset('../../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.pre.dat.nc')
    fGpr = xr.open_dataset('../../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
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
    fmask = xr.open_dataset('../../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Read data from pickle file
    with open('../../../02.ann.BiasCorrection/9-non/China_his_temp_weightmean.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his = \
       data_dict['bigMax_ann_his_weight'], data_dict['bigMin_ann_his_weight']

    with open('../../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = data_dict['bigMax_all'], data_dict['bigMin_all']
    
    with open('../../../02.ann.BiasCorrection/9-non/China_his_temp_weightmean_else.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his, bigMin_ccsm_his = \
        data_dict['bigMax_ccsm_tr'], data_dict['bigMin_ccsm_tr']

    bigMax_ann_his, bigMin_ann_his = \
        monthlymean(bigMax_ann_his), monthlymean(bigMin_ann_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ccsm_his), monthlymean(bigMin_ccsm_his)

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    bigMax_cru_his, bigMin_cru_his = \
        weight_mean_cru(bigMax_cru_his), weight_mean_cru(bigMin_cru_his)

    bigMax_ann_his, bigMin_ann_his, bigMax_cru_his, bigMin_cru_his, bigMax_ccsm_his, bigMin_ccsm_his= \
        bigMax_ann_his.reshape(-1) - 273.15, bigMin_ann_his.reshape(-1) - 273.15, \
        bigMax_cru_his.reshape(-1), bigMin_cru_his.reshape(-1), \
        bigMax_ccsm_his.reshape(-1) - 273.15, bigMin_ccsm_his.reshape(-1) - 273.15

    with open('China_talyor_ccsm_9non.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_his': bigMax_ann_his, 
                     'bigMin_ann_his': bigMin_ann_his, 
                     'bigMax_ccsm_his': bigMax_ccsm_his, 
                     'bigMin_ccsm_his': bigMin_ccsm_his, 
                     'bigMax_cru_his': bigMax_cru_his,
                     'bigMin_cru_his': bigMin_cru_his}, f, protocol=4)





