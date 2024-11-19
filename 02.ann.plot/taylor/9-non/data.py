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


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Read data from pickle file
    with open('../../../02.ann.BiasCorrection/9-non/Beijing_his_temp_postwork.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his = \
       data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    with open('../../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = data_dict['bigMax_all'], data_dict['bigMin_all']
    
    with open('../../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']
    
    with open('../../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    bigMax_ann_his, bigMin_ann_his = \
        monthlymean(bigMax_ann_his), monthlymean(bigMin_ann_his)

    bigMax_ann_his, bigMin_ann_his = \
        bigMax_ann_his[20:50, :], bigMin_ann_his[20:50, :]

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his[:, 26, 37], bigMin_ccsm_his[:, 26, 37]

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his.values.reshape(56, 365), bigMin_ccsm_his.values.reshape(56, 365)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ccsm_his), monthlymean(bigMin_ccsm_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his[20:50, :], bigMin_ccsm_his[20:50, :]

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his[20:50, :, 259, 592], bigMin_cru_his[20:50, :, 259, 592]

    bigMax_ann_his, bigMin_ann_his, bigMax_cru_his, bigMin_cru_his, bigMax_ccsm_his, bigMin_ccsm_his= \
        bigMax_ann_his.reshape(-1) - 273.15, bigMin_ann_his.reshape(-1) - 273.15, \
        bigMax_cru_his.reshape(-1), bigMin_cru_his.reshape(-1), \
        bigMax_ccsm_his.reshape(-1) - 273.15, bigMin_ccsm_his.reshape(-1) - 273.15

    with open('Beijing_talyor_ccsm_9non.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_his': bigMax_ann_his, 
                     'bigMin_ann_his': bigMin_ann_his, 
                     'bigMax_ccsm_his': bigMax_ccsm_his, 
                     'bigMin_ccsm_his': bigMin_ccsm_his, 
                     'bigMax_cru_his': bigMax_cru_his,
                     'bigMin_cru_his': bigMin_cru_his}, f, protocol=4)





