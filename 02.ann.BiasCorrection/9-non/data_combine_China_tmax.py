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


def monthlymean(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    nd_ = bigArray.size
    ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


def combine_data(big_ann_tr, big_ann_pr):
    big_ann_new = np.zeros((56, 365, Nlat, Nlon))
    c = 0
    cl = -1
    for i in range(56):
        if i % 4 ==0 or i % 4 == 1 or i % 4 == 2:
            cl = cl+1
            big_ann_new[i, :, :, :] = big_ann_tr[cl, :, :, :]
        elif i % 4 == 3:
            c = c+1
            cc = c*3
            j = i - cc
            big_ann_new[i, :, :, :] = big_ann_pr[j, :, :, :]
    return big_ann_new


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Load Data
    with open('../../01.ann.train/China_ann_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_tr, bigMax_ann_pr, bigMin_ann_tr, bigMin_ann_pr = \
      data_dict['bigMax_ann_tr'], data_dict['bigMax_ann_pr'], \
      data_dict['bigMin_ann_tr'], data_dict['bigMin_ann_pr']

    bigMax_ann_tr, bigMax_ann_pr = \
        bigMax_ann_tr.reshape(42, 365, Nlat, Nlon), bigMax_ann_pr.reshape(14, 365, Nlat, Nlon)

    bigMin_ann_tr, bigMin_ann_pr = \
        bigMin_ann_tr.reshape(42, 365, Nlat, Nlon), bigMin_ann_pr.reshape(14, 365, Nlat, Nlon)

    bigMax_ann_new, bigMin_ann_new = \
        combine_data(bigMax_ann_tr, bigMax_ann_pr), combine_data(bigMin_ann_tr, bigMin_ann_pr)

    with open('China_historical_tmax_raw.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_historical': bigMax_ann_new}, f, protocol=4)
                     #'bigMin_ann_historical': bigMin_ann_new}, f, protocol=4)




