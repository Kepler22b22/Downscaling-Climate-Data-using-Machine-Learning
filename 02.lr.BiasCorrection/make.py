# Zhang MQ, Mar 2020

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


def monthlymean(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    ny_, nd_ = bigArray.shape
    #ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    #bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


def weight_mean(big_lr_his):
    #big_ann_historical = big_ann_historical.reshape(56, 365, Nlat, Nlon)
    w = np.zeros((Nlat))
    big_lr_new = np.zeros((56, 365, Nlat, Nlon))
    total_w = 0.0
    for j in range(Nlat):
        clat = math.radians(lat[j])
        w[j] = np.cos(clat)
        big_lr_new[:, :, j, :] = big_lr_his[:, :, j, :] * w[j]
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                lat_ = math.radians(lat[j])
                total = np.cos(lat_)
                total_w = total_w + total
    big_lr_new = big_lr_new.reshape(56, 365, 160*280)
    big_lr_new = np.nansum(big_lr_new, axis = 2)
    big_lr_new = big_lr_new / total_w
    return big_lr_new

if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    #Load data
    with open('../01.lr.train/China_his_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_lr_his, bigMin_lr_his = \
        data_dict['bigMax_linear_his'], data_dict['bigMin_linear_his']

    bigMax_lr_his, bigMin_lr_his = bigMax_lr_his[:, :, 99, 185], bigMin_lr_his[:, :, 99, 185]
    bigMax_lr_his, bigMin_lr_his = monthlymean(bigMax_lr_his), monthlymean(bigMin_lr_his)

    with open('Beijing_his_temp.pkl', 'wb') as f:
        pickle.dump({'bigMax_lr_his': bigMax_lr_his, 
                     'bigMin_lr_his': bigMin_lr_his}, f, protocol = 4)








