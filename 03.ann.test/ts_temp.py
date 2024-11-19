# Zhang MQ, sep 2019

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
    bigMax_pr_rcp26 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    bigMin_pr_rcp26 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    bigMax_pr_rcp85 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    bigMin_pr_rcp85 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)
                print("%s: In China！！j=%i, i=%i" % (dataid, j, i))
                
                fname = "./out_temp_rcp26/ann_data_temp_rcp26." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_pr = f['data_predict'][1]
		# 2. put into Big array
                bigMax_pr_rcp26[:, j, i] = data_pr[:, 0]
                bigMin_pr_rcp26[:, j, i] = data_pr[:, 1]

                fname = "./out_temp_rcp85/ann_data_temp_rcp85." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_pr = f['data_predict'][1]
		# 2. put into Big array
                bigMax_pr_rcp85[:, j, i] = data_pr[:, 0]
                bigMin_pr_rcp85[:, j, i] = data_pr[:, 1]
                #print('RCP85 Done. ')

    return bigMax_pr_rcp26, bigMin_pr_rcp26, bigMax_pr_rcp85, bigMin_pr_rcp85


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Combine
    bigMax_pr_rcp26, bigMin_pr_rcp26, bigMax_pr_rcp85, bigMin_pr_rcp85 = load_data()

    # Dump and Load
    with open('China_ann_temp.pkl', 'wb') as f:
        pickle.dump({'bigMax_pr_rcp26': bigMax_pr_rcp26, 
                     'bigMin_pr_rcp26': bigMin_pr_rcp26,
                     'bigMax_pr_rcp85': bigMax_pr_rcp85,
                     'bigMin_pr_rcp85': bigMin_pr_rcp85}, f, protocol = 4)




