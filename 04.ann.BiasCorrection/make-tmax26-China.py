# Zhang MQ, Feb 2020

import os
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


def postwork(alpha, big_ann_tr, big_ann_pr, big_ground_tr):
    ny_, nd_, nlat_, nlon_ = big_ann_pr.shape
    big_ann = np.zeros((ny_, 365, Nlat, Nlon))
    for j in range(Nlat):
        for i in range(Nlon):
            for l in range(9):
                for k in range(12):
                    if m[j, i]:
                        mean_ann_tr = np.mean(big_ann_tr[:, m_aa[k]:m_bb[k], j, i])
                        mean_ann_pr = np.mean(big_ann_pr[n_aa[l]:n_bb[l], m_aa[k]:m_bb[k], j, i])
                        mean_ground_tr = np.mean(big_ground_tr[:, m_aa[k]:m_bb[k], j, i])
                        big_ann[n_aa[l]:n_bb[l], m_aa[k]:m_bb[k], j, i] = ((big_ann_pr[n_aa[l]:n_bb[l], m_aa[k]:m_bb[k], j, i] - \
                                                                            mean_ann_pr) * alpha[k, j, i]) + \
                                                                          (mean_ann_pr - mean_ann_tr + mean_ground_tr)
                    else:
                        big_ann[n_aa[l]:n_bb[l], m_aa[k]:m_bb[k], j, i] = np.nan
    return big_ann


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    n_aa = [0,  15, 25, 35, 45, 55, 65, 75, 85]
    n_bb = [15, 25, 35, 45, 55, 65, 75, 85, 95]

    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # Load Data
    with open('../02.ann.BiasCorrection/9-non/China_historical_tmax_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his = data_dict['bigMax_ann_historical']
 
    with open('../03.ann.test/China_ann_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_rcp26, bigMax_ann_rcp85 = \
        data_dict['bigMax_pr_rcp26'], data_dict['bigMax_pr_rcp85']

    with open('../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ground_his = data_dict['bigMax_ground_tr']

    with open('../02.ann.BiasCorrection/9-non/China_his_temp_postwork.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    alpha_tmax = data_dict['alpha_tmax']
    
    print('Data Loaded. ')

    bigMax_ann_rcp26 = \
        bigMax_ann_rcp26.reshape(95, 365, Nlat, Nlon)

    bigMax_ground_his = bigMax_ground_his.values.reshape(56, 365, Nlat, Nlon)

    bigMax_ann_rcp26 = \
        postwork(alpha_tmax, bigMax_ann_his, bigMax_ann_rcp26, bigMax_ground_his), \

    with open('China_tmax_post_rcp26.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_rcp26': bigMax_ann_rcp26}, f, protocol=4)



