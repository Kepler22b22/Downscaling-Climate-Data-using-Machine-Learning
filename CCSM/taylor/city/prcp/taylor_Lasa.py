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
    with open('../../../01.bc.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his = data_dict['bigPr_bc_his']

    with open('../../../../../Downscaling-2020-10/CCSM_case/01.ann.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ann_his = data_dict['bigPr_ann_his']

    with open('../../../../../Downscaling-2020-10/CCSM_case/01.lr.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_lr_his = data_dict['bigPr_linear_his']

    with open('../../../../../Downscaling-2020-10/CCSM_case/01.poisson.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_poisson_his = data_dict['bigPr_poisson_his']

    with open('../../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_cru_his = data_dict['bigPr_all']
    
    with open('../../../../data/ccsm/future/tr_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_his = data_dict['bigPr_ccsm_tr']
    
    bigPr_bc_his = bigPr_bc_his[20:50, :, 59, 84]
    bigPr_bc_his = monthlymean(bigPr_bc_his)

    bigPr_ann_his = bigPr_ann_his[20:50, :, 59, 84]
    bigPr_ann_his = monthlymean(bigPr_ann_his)

    bigPr_lr_his = bigPr_lr_his[20:50, :, 59, 84]
    bigPr_lr_his = monthlymean(bigPr_lr_his)

    bigPr_poisson_his = bigPr_poisson_his[20:50, :, 59, 84]
    bigPr_poisson_his = monthlymean(bigPr_poisson_his)

    bigPr_ccsm_his = bigPr_ccsm_his[:, 15, 17]
    bigPr_ccsm_his = bigPr_ccsm_his.values.reshape(56, 365)
    bigPr_ccsm_his = monthlymean(bigPr_ccsm_his)
    bigPr_ccsm_his = bigPr_ccsm_his[20:50, :]

    bigPr_cru_his = bigPr_cru_his.values.reshape(56, 12, 360, 720)
    bigPr_cru_his = bigPr_cru_his[20:50, :, 239, 542]
    mm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(12):
        bigPr_cru_his[:, i] = bigPr_cru_his[:, i] / mm[i]

    bigPr_bc_his, bigPr_ann_his, bigPr_lr_his, bigPr_poisson_his, bigPr_cru_his, bigPr_ccsm_his= \
        bigPr_bc_his.reshape(-1) * 86400, \
        bigPr_ann_his.reshape(-1) * 86400, \
        bigPr_lr_his.reshape(-1) * 86400, \
        bigPr_poisson_his.reshape(-1) * 86400, \
        bigPr_cru_his.reshape(-1), \
        bigPr_ccsm_his.reshape(-1) * 86400

    bigPr_ann_his = bigPr_ann_his / 1e7
    bigPr_lr_his = bigPr_lr_his / 1e7

    with open('Lasa_talyor_ccsm.pkl', 'wb') as f:
        pickle.dump({'bigPr_bc_his': bigPr_bc_his, 
                     'bigPr_ann_his': bigPr_ann_his, 
                     'bigPr_lr_his': bigPr_lr_his, 
                     'bigPr_poisson_his': bigPr_poisson_his, 
                     'bigPr_ccsm_his': bigPr_ccsm_his, 
                     'bigPr_cru_his': bigPr_cru_his}, f, protocol=4)




