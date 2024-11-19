# Zhang MQ, Dec 2019

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


def load_data():
    bigMax_ann_tr = np.zeros((15330, Nlat, Nlon), dtype='float32')
    bigMax_ann_pr = np.zeros((5110, Nlat, Nlon), dtype='float32')
    bigMin_ann_tr = np.zeros((15330, Nlat, Nlon), dtype='float32')
    bigMin_ann_pr = np.zeros((5110, Nlat, Nlon), dtype='float32')
    
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)
                print("%s: In China！！j=%i, i=%i" % (dataid, j, i))
                fname = "./out_temp1/ann_data_temp." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_ann_pr = f['data_predict'][2]
                    data_ann_tr = f['data_train'][2]
		# 2. put into Big array
                bigMax_ann_tr[:, j, i] = data_ann_tr[:, 0]
                bigMax_ann_pr[:, j, i] = data_ann_pr[:, 0]
                bigMin_ann_tr[:, j, i] = data_ann_tr[:, 1]
                bigMin_ann_pr[:, j, i] = data_ann_pr[:, 1]

    return bigMax_ann_tr, bigMax_ann_pr, bigMin_ann_tr, bigMin_ann_pr


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Combine
    bigMax_ann_tr, bigMax_ann_pr, bigMin_ann_tr, bigMin_ann_pr = load_data()

    # Dump and Load
    with open('China_ann_temp1.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_tr': bigMax_ann_tr, 
                     'bigMax_ann_pr': bigMax_ann_pr, 
                     'bigMin_ann_tr': bigMin_ann_tr,
                     'bigMin_ann_pr': bigMin_ann_pr}, f)

    # Load Data
    #with open('China_ann_temp.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ann_tr, bigMax_ann_pr, bigMin_ann_tr, bigMin_ann_pr = \
    #  data_dict['bigMax_ann_tr'], data_dict['bigMax_ann_pr'], \
    #  data_dict['bigMin_ann_tr'], data_dict['bigMin_ann_pr']




