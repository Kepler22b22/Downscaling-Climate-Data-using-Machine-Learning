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


def weight_mean(big_ann):
    #big_ann_historical = big_ann_historical.reshape(56, 365, Nlat, Nlon)
    w = np.zeros((Nlat))
    big_ann_new = np.zeros((95, 365, Nlat, Nlon))
    total_w = 0.0
    for j in range(Nlat):
        clat = math.radians(lat[j])
        w[j] = np.cos(clat)
        big_ann_new[:, :, j, :] = big_ann[:, :, j, :] * w[j]
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                lat_ = math.radians(lat[j])
                total = np.cos(lat_)
                total_w = total_w + total
    big_ann_new = big_ann_new.reshape(95, 365, 160*280)
    big_ann_new = np.nansum(big_ann_new, axis = 2)
    big_ann_new = big_ann_new / total_w
    return big_ann_new

if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    #Load data
    with open('./China_tmax_post_rcp26.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_rcp26 = data_dict['bigMax_ann_rcp26']
    
    bigMax_ann_rcp26 = np.squeeze(np.array(bigMax_ann_rcp26))
    bigMax_ann_rcp26 = weight_mean(bigMax_ann_rcp26)

    with open('China_rcp26_tmax_weightmean.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_rcp26_weight': bigMax_ann_rcp26}, f, protocol = 4)





