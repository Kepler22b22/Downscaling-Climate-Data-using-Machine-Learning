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


def get_alpha(big_ann_tr, big_ground_tr):
    alpha_new = np.zeros((12, Nlat, Nlon))
    #big_ann_tr = big_ann_tr.reshape(42, 365, Nlat, Nlon)
    #big_ground_tr = big_ground_tr.reshape(42, 365, Nlat, Nlon)
    big_ann_tr[:, :, 1 - m.values.astype(np.int)] = np.nan
    big_ground_tr[:, :, 1 - m.values.astype(np.int)] = np.nan
    for j in range(Nlat):
        for i in range(Nlon): 
            for k in range(12):
                big_ann = big_ann_tr[:, m_aa[k]:m_bb[k], j, i].reshape(-1)
                big_ground = big_ground_tr[:, m_aa[k]:m_bb[k], j, i].reshape(-1)
                alpha_new[k, j, i] = np.std(big_ground) / np.std(big_ann)
    return alpha_new


def postwork(alpha, big_ann_tr, big_ann_pr, big_ground_tr):
    ny_, nd_, nlat_, nlon_ = big_ann_pr.shape
    big_ann = np.zeros((ny_, 365, Nlat, Nlon))
    for j in range(Nlat):
        for i in range(Nlon):
            for k in range(12):
                if m[j, i]:
                    mean_ann_tr = np.mean(big_ann_tr[:, m_aa[k]:m_bb[k], j, i])
                    mean_ann_pr = np.mean(big_ann_pr[:, m_aa[k]:m_bb[k], j, i])
                    mean_ground_tr = np.mean(big_ground_tr[:, m_aa[k]:m_bb[k], j, i])
                    big_ann[:, m_aa[k]:m_bb[k], j, i] = ((big_ann_pr[:, m_aa[k]:m_bb[k], j, i] - mean_ann_pr) * alpha[k, j, i]) + \
                                                        (mean_ann_pr - mean_ann_tr + mean_ground_tr)
                else:
                    big_ann[:, m_aa[k]:m_bb[k], j, i] = np.nan
    return big_ann


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # Load Data
    with open('./China_historical_tmax_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his = \
      data_dict['bigMax_ann_historical']
 
    with open('./China_historical_tmin_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ann_his = \
      data_dict['bigMin_ann_historical']

    with open('../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ground_his = data_dict['bigMax_ground_tr']

    with open('../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ground_his = data_dict['bigMin_ground_tr']

    print('Data Loaded. ')

    bigMax_ann_his, bigMin_ann_his = \
        bigMax_ann_his.reshape(56, 365, Nlat, Nlon), bigMin_ann_his.reshape(56, 365, Nlat, Nlon)

    bigMax_ground_his, bigMin_ground_his = \
        bigMax_ground_his.values.reshape(56, 365, Nlat, Nlon), bigMin_ground_his.values.reshape(56, 365, Nlat, Nlon)

    alpha_tmax = get_alpha(bigMax_ann_his, bigMax_ground_his)
    alpha_tmin = get_alpha(bigMin_ann_his, bigMin_ground_his)

    bigMax_ann_his, bigMin_ann_his = \
        postwork(alpha_tmax, bigMax_ann_his, bigMax_ann_his, bigMax_ground_his), \
        postwork(alpha_tmin, bigMin_ann_his, bigMin_ann_his, bigMin_ground_his)

    with open('China_his_temp_postwork.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_his': bigMax_ann_his,
                     'bigMin_ann_his': bigMin_ann_his,
                     'alpha_tmax': alpha_tmax,
                     'alpha_tmin': alpha_tmin}, f, protocol=4)




