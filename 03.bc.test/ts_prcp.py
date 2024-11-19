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
    bigPr_pr_rcp26 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    bigPr_pr_rcp85 = np.zeros((34675, Nlat, Nlon), dtype='float32')
    
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)
                print("%s: In China！！j=%i, i=%i" % (dataid, j, i))
                
                fname = "./out_pr_rcp26/bc_data_pr_rcp26." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_pr = f['data_predict'][1]
		# 2. put into Big array
                bigPr_pr_rcp26[:, j, i] = data_pr[:]
                print('RCP26 Done. ')

                fname = "./out_pr_rcp85/bc_data_pr_rcp85." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_pr = f['data_predict'][1]
		# 2. put into Big array
                bigPr_pr_rcp85[:, j, i] = data_pr[:]
                print('RCP85 Done. ')

    return bigPr_pr_rcp26, bigPr_pr_rcp85


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Combine
    bigPr_pr_rcp26, bigPr_pr_rcp85 = load_data()

    # Dump and Load
    with open('China_bc_prcp.pkl', 'wb') as f:
        pickle.dump({'bigPr_pr_rcp26': bigPr_pr_rcp26, 
                     'bigPr_pr_rcp85': bigPr_pr_rcp85}, f, protocol = 4)




