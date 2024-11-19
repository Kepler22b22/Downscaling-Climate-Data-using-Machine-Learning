# Zhang MQ, Feb 2020

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
    ny_, nd_, nlon_, nlat_ = bigArray.shape
    #ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12, nlon_, nlat_), dtype='float32')
    #bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i, :, :] = np.mean(bigArray[:, m_aa[i]:m_bb[i], :, :], axis=1)
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
    with open('../02.ann.BiasCorrection/China_his_temp_postwork.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his = \
       data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    with open('../../data/ccsm/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']
    
    with open('../../data/ccsm/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    with open('../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = \
        data_dict['bigMax_all'], data_dict['bigMin_all']

    print(bigMax_cru_his.shape)
    print(bigMax_ann_his.shape)
    print(bigMax_ccsm_his.shape)

    bigMax_ann_his, bigMin_ann_his, bigMax_ccsm_his, bigMin_ccsm_his, bigMax_cru_his, bigMin_cru_his = \
        bigMax_ann_his.reshape(56, 365, 160, 280), bigMin_ann_his.reshape(56, 365, 160, 280), \
        bigMax_ccsm_his.values.reshape(56, 365, 42, 57), bigMin_ccsm_his.values.reshape(56, 365, 42, 57), \
        bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    bigMax_ann_his, bigMin_ann_his, bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ann_his) - 273.15, monthlymean(bigMin_ann_his) - 273.15, \
        monthlymean(bigMax_ccsm_his) - 273.15, monthlymean(bigMin_ccsm_his) - 273.15

    print('Data Loaded. ')

    # Time series
    print('Now Doing Time Series...')
    
    # July
    regions = ['eNWC', 'wNWC', 'Plateau', 'NC', 'NEC', 
            'nSWC', 'sSWC', 'CC', 'EC', 'SC']

    latA1 = []
    latA2 = []
    lonA1 = []
    lonA2 = []

    latG1 = [252, 220, 252, 252, 260, 234, 210, 234, 234, 210]
    latG2 = [280, 252, 280, 280, 290, 252, 234, 252, 252, 234]
    lonG1 = [550, 500, 500, 582, 598, 556, 550, 572, 592, 572]
    lonG2 = [582, 556, 550, 598, 630, 572, 572, 592, 630, 630]

    latC = [32, 28, 22, 20, 17, 8, 16, 15, 30, 22, 26]
    lonC = [45, 43, 40, 35, 41, 35, 27, 17, 14, 27, 37]

    for i in range(0, 11):
        ytr = bigMax_cru_his[20:50, 6, latG[i], lonG[i]].reshape(-1)
        yCtr = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]].reshape(-1)
        yACtr = bigMax_ann_his[20:50, 6, latA[i], lonA[i]].reshape(-1)

        x_array = np.arange(1970, 2000)

        minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
        maxX = max(max(ytr), max(yCtr), max(yACtr)) + 1

        plot2d(3, x=[x_array, x_array, x_array], 
               y=[ytr, yCtr, yACtr],
               style=['r-o', 'k-o', 'b-o'], 
               lw=[2.5, 1.7, 2],
               lb=['CRU', 'CCSM', 'SD'],
               ti='(b) Tmax (July)' + '\n' + city[i],
               xl='Year', yl='Tmax (Celsius)',
               xlim=[x_array[0], x_array[-1]],
               ylim=[minX, maxX, 1],
               figsize=(8, 4),
               fn='./figures/plot-tmax-China-historical-' + city[i] + '.pdf')

    # Jan





    #ytr = bigMin_cru_his[20:50, 0].reshape(-1)
    #yCtr = bigMin_ccsm_his[20:50, 0].reshape(-1)
    #yACtr = bigMin_ann_his[20:50, 0].reshape(-1)

    #x_array = np.arange(1970, 2000)

    #plot2d(2, x=[x_array, x_array], 
    #       y=[ytr,yCtr],
    #       style=['r-o', 'k-o'], 
    #       lw=[2.5, 1.7],
    #       lb=['CRU', 'CCSM'],
    #       ti='(a) Tmin (January)',
    #       xl='Year', yl='Tmin (Celsius)',
    #       xlim=[x_array[0], x_array[-1]],
    #       ylim=[-19, -11, 1],
    #       figsize=(8, 4),
    #       fn='./figures/plot-tmin-China-historical.pdf')













