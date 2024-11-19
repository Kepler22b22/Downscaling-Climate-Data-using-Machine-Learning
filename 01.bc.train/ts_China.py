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
    bigPr_bc_tr = np.zeros((15330, Nlat, Nlon), dtype='float32')
    bigPr_bc_pr = np.zeros((5110, Nlat, Nlon), dtype='float32')
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)
                print("%s: In China！！j=%i, i=%i" % (dataid, j, i))
                fname = "./out/bc_data." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_ann)
                    data_bc_pr = f['data_predict'][2]
                    data_bc_tr = f['data_train'][2]
		# 2. put into Big array
                bigPr_bc_tr[:, j, i] = data_bc_tr[:]
                bigPr_bc_pr[:, j, i] = data_bc_pr[:]

    return bigPr_bc_tr, bigPr_bc_pr


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Combine
    #bigPr_bc_tr, bigPr_bc_pr = load_data()

    # Dump and Load
    #with open('China_bc_prcp.pkl', 'wb') as f:
    #    pickle.dump({'bigPr_bc_tr': bigPr_bc_tr, 
    #                 'bigPr_bc_pr': bigPr_bc_pr}, f)

    with open('./China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his = data_dict['bigPr_bc_his']

    print(bigPr_bc_his[0, 0, 99, 185])




