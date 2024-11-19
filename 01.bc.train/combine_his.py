# Zhang MQ, Jun 2020

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
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Load Data
    with open('./China_bc_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_tr, bigPr_bc_pr = \
      data_dict['bigPr_bc_tr'], data_dict['bigPr_bc_pr']
    
    bigPr_bc_tr, bigPr_bc_pr = \
        bigPr_bc_tr.reshape(42, 365, Nlat, Nlon), bigPr_bc_pr.reshape(14, 365, Nlat, Nlon)

    bigPr_bc_new = combine_data(bigPr_bc_tr, bigPr_bc_pr)

    with open('China_his_prcp.pkl', 'wb') as f:
        pickle.dump({'bigPr_bc_his': bigPr_bc_new}, f, protocol=4)








