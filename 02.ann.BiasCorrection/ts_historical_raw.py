# Zhang MQ, Mar 2021

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
    nd_ = bigArray.size
    ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


def plot2d(N, x, y, style, lw, lb, 
           ti='Plot', xl='X', yl='Y', legendloc=4, bbox_2_anchor=(1.1434, 0.1), 
           xlim=[], ylim=[], ylog=False,
           figsize=(30, 18),
           fn="plot2d.pdf"):

    plt.figure(figsize=figsize)

    for i in range(N):
        plt.plot(x[i], y[i], style[i], linewidth=lw[i], label=lb[i], alpha=0.6)

    plt.title(ti)
    plt.xlabel(xl)
    plt.ylabel(yl)

    if xlim: plt.xlim(xlim[0], xlim[1])
    if ylim: plt.ylim(ylim[0], ylim[1])
    if ylog: plt.yscale('log')
    plt.xticks([1973, 1977, 1981, 1985, 1989, 1993, 1997])

    plt.legend(loc=legendloc)
    plt.savefig(fn)
    plt.close()
    print('Figure was saved into file: ' + fn)
    return


if __name__ == '__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # Load Data
    with open('./9-non/China_his_temp_weightmean_else.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_raw, bigMin_ann_raw = \
        data_dict['bigMax_ann_raw'], data_dict['bigMin_ann_raw']

    with open('../02.ann.plot/taylor/9-non/China_talyor_ccsm_9non.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his, bigMin_ccsm_his, bigMax_cru_his, bigMin_cru_his = \
        data_dict['bigMax_ccsm_his'], data_dict['bigMin_ccsm_his'], \
        data_dict['bigMax_cru_his'], data_dict['bigMin_cru_his']

    bigMax_ann_raw, bigMin_ann_raw = \
        monthlymean(bigMax_ann_raw) - 273.15, monthlymean(bigMin_ann_raw) - 273.15

    bigMax_ann_raw, bigMin_ann_raw, bigMax_ccsm_his, bigMin_ccsm_his, bigMax_cru_his, bigMin_cru_his = \
        bigMax_ann_raw.reshape(56, 12), bigMin_ann_raw.reshape(56, 12), \
        bigMax_ccsm_his.reshape(56, 12), bigMin_ccsm_his.reshape(56, 12), \
        bigMax_cru_his.reshape(56, 12), bigMin_cru_his.reshape(56, 12)

    print('Data Loaded. ')

    # Time series
    print('Now Doing Time Series...')
    
    # July
    ytr = bigMax_cru_his[20:50, 6].reshape(-1)
    yCtr = bigMax_ccsm_his[20:50, 6].reshape(-1)
    yACtr = bigMax_ann_raw[20:50, 6].reshape(-1)

    x_array = np.arange(1970, 2000)

    mean_ccsm = np.mean(yCtr)
    mean_cru = np.mean(ytr)   
    mean_ccsm_ann = np.mean(yACtr)
  
    print('mean_ccsm = ', mean_ccsm)
    print('mean_ccsm_ann = ', mean_ccsm_ann)
    print('mean_cru = ', mean_cru)

    std_ccsm = np.std(yCtr)
    std_cru = np.std(ytr)
    std_ccsm_ann = np.std(yACtr)
    
    print('std_ccsm = ', std_ccsm)
    print('std_ccsm_ann = ', std_ccsm_ann)
    print('std_cru = ', std_cru)

    mse_ccsm = mean_squared_error(ytr, yCtr)
    rmse_ccsm = np.sqrt(mse_ccsm)    
    mse_ccsm_ann = mean_squared_error(ytr, yACtr)
    rmse_ccsm_ann = np.sqrt(mse_ccsm_ann)

    print('rmse_ccsm = ', rmse_ccsm)
    print('rmse_ccsm_ann = ', rmse_ccsm_ann)

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU', 'CCSM', 'SD'],
           ti='Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[22.5, 26.5, 1],
           figsize=(8, 4),
           fn='./figures/plot-tmax-China-historical-raw.pdf')

    # Jan
    ytr = bigMin_cru_his[20:50, 0].reshape(-1)
    yCtr = bigMin_ccsm_his[20:50, 0].reshape(-1)
    yACtr = bigMin_ann_raw[20:50, 0].reshape(-1)

    x_array = np.arange(1970, 2000)

    mean_ccsm = np.mean(yCtr)
    mean_cru = np.mean(ytr)
    mean_ccsm_ann = np.mean(yACtr)
  
    print('mean_ccsm = ', mean_ccsm)
    print('mean_ccsm_ann = ', mean_ccsm_ann)
    print('mean_cru = ', mean_cru)

    std_ccsm = np.std(yCtr)
    std_cru = np.std(ytr)
    std_ccsm_ann = np.std(yACtr)
    
    print('std_ccsm = ', std_ccsm)
    print('std_ccsm_ann = ', std_ccsm_ann)
    print('std_cru = ', std_cru)

    mse_ccsm = mean_squared_error(ytr, yCtr)
    rmse_ccsm = np.sqrt(mse_ccsm)
    mse_ccsm_ann = mean_squared_error(ytr, yACtr)
    rmse_ccsm_ann = np.sqrt(mse_ccsm_ann)

    print('rmse_ccsm = ', rmse_ccsm)
    print('rmse_ccsm_ann = ', rmse_ccsm_ann)

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr,yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU', 'CCSM', 'ANN-9p'],
           ti='Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[-19, -10, 1],
           figsize=(8, 4),
           fn='./figures/plot-tmin-China-historical-raw.pdf')

