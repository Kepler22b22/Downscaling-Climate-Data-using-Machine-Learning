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
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Read data from pickle file
    with open('../../02.ann.BiasCorrection/China_his_temp_postwork_9non.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his = \
       data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    with open('../../02.ann.BiasCorrection/China_his_temp_postwork_1non.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his_1non, bigMin_ann_his_1non = \
        data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    with open('../../../../Downscaling-2020-10/CCSM_case/01.lr.train/China_his_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_lr_his, bigMin_lr_his = \
        data_dict['bigMax_linear_his'], data_dict['bigMin_linear_his']

    with open('../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = data_dict['bigMax_all'], data_dict['bigMin_all']
    
    with open('../../../data/ccsm/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']
    
    with open('../../../data/ccsm/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    bigMax_ann_his, bigMin_ann_his = \
        bigMax_ann_his[20:50, :, 123, 226], bigMin_ann_his[20:50, :, 123, 226]

    bigMax_ann_his, bigMin_ann_his = \
        monthlymean(bigMax_ann_his), monthlymean(bigMin_ann_his)

    bigMax_ann_his_1non, bigMin_ann_his_1non = \
        bigMax_ann_his_1non[20:50, :, 123, 226], bigMin_ann_his_1non[20:50, :, 123, 226]

    bigMax_ann_his_1non, bigMin_ann_his_1non = \
        monthlymean(bigMax_ann_his_1non), monthlymean(bigMin_ann_his_1non)

    bigMax_lr_his, bigMin_lr_his = \
        bigMax_lr_his[20:50, :, 123, 226], bigMin_lr_his[20:50, :, 123, 226]

    bigMax_lr_his, bigMin_lr_his = \
        monthlymean(bigMax_lr_his), monthlymean(bigMin_lr_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his[:, 32, 45], bigMin_ccsm_his[:, 32, 45]

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his.values.reshape(56, 365), bigMin_ccsm_his.values.reshape(56, 365)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ccsm_his), monthlymean(bigMin_ccsm_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his[20:50, :], bigMin_ccsm_his[20:50, :]

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his[20:50, :, 271, 613], bigMin_cru_his[20:50, :, 271, 613]

    bigMax_ann_his, bigMin_ann_his, bigMax_ann_his_1non, bigMin_ann_his_1non, bigMax_lr_his, bigMin_lr_his, bigMax_cru_his, bigMin_cru_his, bigMax_ccsm_his, bigMin_ccsm_his= \
        bigMax_ann_his.reshape(-1) - 273.15, bigMin_ann_his.reshape(-1) - 273.15, \
        bigMax_ann_his_1non.reshape(-1) - 273.15, bigMin_ann_his_1non.reshape(-1) - 273.15, \
        bigMax_lr_his.reshape(-1) - 273.15, bigMin_lr_his.reshape(-1) - 273.15, \
        bigMax_cru_his.reshape(-1), bigMin_cru_his.reshape(-1), \
        bigMax_ccsm_his.reshape(-1) - 273.15, bigMin_ccsm_his.reshape(-1) - 273.15

    with open('Harbin_talyor_ccsm.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_his': bigMax_ann_his, 
                     'bigMin_ann_his': bigMin_ann_his, 
                     'bigMax_ann_his_1non': bigMax_ann_his_1non, 
                     'bigMin_ann_his_1non': bigMin_ann_his_1non, 
                     'bigMax_lr_his': bigMax_lr_his,
                     'bigMin_lr_his': bigMin_lr_his,
                     'bigMax_ccsm_his': bigMax_ccsm_his, 
                     'bigMin_ccsm_his': bigMin_ccsm_his, 
                     'bigMax_cru_his': bigMax_cru_his,
                     'bigMin_cru_his': bigMin_cru_his}, f, protocol=4)





