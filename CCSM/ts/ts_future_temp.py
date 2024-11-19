# Zhang MQ, Apr 2021

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


def smooth(bigArray, WSZ):
    out0 = np.convolve(bigArray, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r= np.arange(1, WSZ-1, 2)
    start = np.cumsum(bigArray[:WSZ-1])[::2] / r
    stop = (np.cumsum(bigArray[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def plot2d(N, x, y, color, linestyle, lw,
           ti='Plot', xl='X', yl='Y', legendloc=4, bbox_2_anchor=(1.1434, 0.1), 
           xlim=[], ylim=[], ylog=False,
           figsize=(30, 18),
           fn="plot2d.pdf"):

    plt.figure(figsize=figsize)

    for i in range(N):
        plt.plot(x[i], y[i], color=color[i], linestyle=linestyle[i], linewidth=lw[i], alpha=0.75)

    plt.title(ti)
    plt.xlabel(xl, fontsize=12)
    plt.ylabel(yl, fontsize=12)

    #if xlim: plt.xlim(xlim[0], xlim[1])
    if ylim: plt.ylim(ylim[0], ylim[1])
    if ylog: plt.yscale('log')
    plt.xticks([0, 10, 20, 29, 30, 40, 50, 61, 62, 72, 82, 92], 
               ['1970', '1980', '1990', '9', '2', '2030', '2040', '0',  '2', ' 2080', '2090', '2100'], fontsize=10)
    #plt.xticks([15, 45, 77], 
    #           ['\nCLIM', '\nNear-term', '\nLong-term'], weight='bold')
    #plt.xticks([0, 10, 20, 29, 30, 40, 50, 61, 62, 72, 82, 92],
    #           ['1970', '1980', '1990', '9', '2', '2030', '2040', '0',  '2', '2080', '2090', '2100'])

    #plt.legend(loc=legendloc, ncol=3) #bbox_to_anchor=bbox_2_anchor)
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
    #with open('../02.ann.BiasCorrection/China_his_temp_postwork_9non.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ann_his, bigMin_ann_his = \
    #   data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    #with open('../../data/ccsm/future/tr_tmax.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']
    
    #with open('../../data/ccsm/future/tr_tmin.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    #with open('../../data/cru/cru_all.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_cru_his, bigMin_cru_his = \
    #    data_dict['bigMax_all'], data_dict['bigMin_all']
    
    #with open('../../../Downscaling-2020-10/CCSM_case/04.ann.BiasCorrection/China_tmax_post_rcp26.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ann_rcp26 = data_dict['bigMax_ann_rcp26']

    #with open('../../../Downscaling-2020-10/CCSM_case/04.ann.BiasCorrection/China_tmax_post_rcp85.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ann_rcp85 = data_dict['bigMax_ann_rcp85']

    #with open('../../../Downscaling-2020-10/CCSM_case/04.ann.BiasCorrection/China_tmin_post_rcp26.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ann_rcp26 = data_dict['bigMin_ann_rcp26']

    #with open('../../../Downscaling-2020-10/CCSM_case/04.ann.BiasCorrection/China_tmin_post_rcp85.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ann_rcp85 = data_dict['bigMin_ann_rcp85']

    #with open('../../../Downscaling-2020-10/data/ccsm4/future/ccsm_rcp_tmax.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ccsm_rcp26, bigMax_ccsm_rcp85 = data_dict['bigMax_ccsm_rcp26'], data_dict['bigMax_ccsm_rcp85']

    #with open('../../../Downscaling-2020-10/data/ccsm4/future/ccsm_rcp_tmin.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ccsm_rcp26, bigMin_ccsm_rcp85 = data_dict['bigMin_ccsm_rcp26'], data_dict['bigMin_ccsm_rcp85']

    #bigMax_ann_his, bigMin_ann_his, bigMax_ccsm_his, bigMin_ccsm_his, bigMax_cru_his, bigMin_cru_his = \
    #    bigMax_ann_his.reshape(56, 365, 160, 280), bigMin_ann_his.reshape(56, 365, 160, 280), \
    #    bigMax_ccsm_his.values.reshape(56, 365, 42, 57), bigMin_ccsm_his.values.reshape(56, 365, 42, 57), \
    #    bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    #bigMax_ann_his, bigMin_ann_his, bigMax_ccsm_his, bigMin_ccsm_his = \
    #    monthlymean(bigMax_ann_his) - 273.15, monthlymean(bigMin_ann_his) - 273.15, \
    #    monthlymean(bigMax_ccsm_his) - 273.15, monthlymean(bigMin_ccsm_his) - 273.15

    #bigMax_ann_rcp26, bigMax_ann_rcp85, bigMin_ann_rcp26, bigMin_ann_rcp85 = \
    #    monthlymean(np.squeeze(np.array(bigMax_ann_rcp26))) - 273.15, \
    #    monthlymean(np.squeeze(np.array(bigMax_ann_rcp85))) - 273.15, \
    #    monthlymean(np.squeeze(np.array(bigMin_ann_rcp26))) - 273.15, \
    #    monthlymean(np.squeeze(np.array(bigMin_ann_rcp85))) - 273.15

    #bigMax_ccsm_rcp26, bigMax_ccsm_rcp85, bigMin_ccsm_rcp26, bigMin_ccsm_rcp85 = \
    #    bigMax_ccsm_rcp26.values.reshape(95, 365, 42, 57), \
    #    bigMax_ccsm_rcp85.values.reshape(95, 365, 42, 57), \
    #    bigMin_ccsm_rcp26.values.reshape(95, 365, 42, 57), \
    #    bigMin_ccsm_rcp85.values.reshape(95, 365, 42, 57)

    #bigMax_ccsm_rcp26, bigMax_ccsm_rcp85, bigMin_ccsm_rcp26, bigMin_ccsm_rcp85 = \
    #    monthlymean(bigMax_ccsm_rcp26) - 273.15, \
    #    monthlymean(bigMax_ccsm_rcp85) - 273.15, \
    #    monthlymean(bigMin_ccsm_rcp26) - 273.15, \
    #    monthlymean(bigMin_ccsm_rcp85) - 273.15

    #with open('China_historical_temp.pkl', 'wb') as f:
    #    pickle.dump({'bigMax_ann_his': bigMax_ann_his,
    #                 'bigMin_ann_his': bigMin_ann_his,
    #                 'bigMax_ccsm_his': bigMax_ccsm_his,
    #                 'bigMin_ccsm_his': bigMin_ccsm_his, 
    #                 'bigMax_cru_his': bigMax_cru_his, 
    #                 'bigMin_cru_his': bigMin_cru_his}, f, protocol=4)
    
    #with open('China_future_temp.pkl', 'wb') as f:
    #    pickle.dump({'bigMax_ann_rcp26': bigMax_ann_rcp26, 
    #                 'bigMin_ann_rcp26': bigMin_ann_rcp26,
    #                 'bigMax_ann_rcp85': bigMax_ann_rcp85, 
    #                 'bigMin_ann_rcp85': bigMin_ann_rcp85, 
    #                 'bigMax_ccsm_rcp26': bigMax_ccsm_rcp26, 
    #                 'bigMin_ccsm_rcp26': bigMin_ccsm_rcp26, 
    #                 'bigMax_ccsm_rcp85': bigMax_ccsm_rcp85, 
    #                 'bigMin_ccsm_rcp85': bigMin_ccsm_rcp85}, f, protocol=4)

    with open('./China_historical_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his, bigMax_ccsm_his, bigMin_ccsm_his, bigMax_cru_his, bigMin_cru_his = \
        data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his'],\
        data_dict['bigMax_ccsm_his'], data_dict['bigMin_ccsm_his'], \
        data_dict['bigMax_cru_his'], data_dict['bigMin_cru_his']

    with open('./China_future_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_rcp26, bigMin_ann_rcp26, bigMax_ann_rcp85, bigMin_ann_rcp85, bigMax_ccsm_rcp26, bigMin_ccsm_rcp26, bigMax_ccsm_rcp85, bigMin_ccsm_rcp85 = \
        data_dict['bigMax_ann_rcp26'], data_dict['bigMin_ann_rcp26'], \
        data_dict['bigMax_ann_rcp85'], data_dict['bigMin_ann_rcp85'], \
        data_dict['bigMax_ccsm_rcp26'], data_dict['bigMin_ccsm_rcp26'], \
        data_dict['bigMax_ccsm_rcp85'], data_dict['bigMin_ccsm_rcp85']

    print('Data Loaded. ')

    # Time series
    print('Now Doing Time Series...')
    
    # July
    region = ['Huabei', 'Dongbei', 'Huadong', 'Huazhong', 
              'Huanan', 'Xinan', 'Xibei']

    #Huabei
    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    latA = [99, 96, 92, 91, 102]
    lonA = [185, 188, 177, 170, 165]

    latG = [259, 258, 256, 255, 261]
    lonG = [592, 594, 588, 585, 582]

    latC = [26, 25, 24, 24, 26]
    lonC = [37, 38, 35, 34, 33]

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Huabei.pdf')

    #Dongbei
    latA = [123, 114, 114, 106, 129]
    lonA = [226, 220, 205, 213, 216]

    latG = [271, 267, 267, 263, 274]
    lonG = [613, 610, 612, 606, 608]

    latC = [32, 30, 30, 28, 34]
    lonC = [45, 44, 45, 43, 43]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Dongbei.pdf')

    #Huadong
    latA = [64, 68, 61, 84, 85]
    lonA = [205, 194, 200, 201, 188]

    latG = [242, 244, 240, 252, 252]
    lonG = [602, 597, 600, 600, 594]

    latC = [17, 17, 16, 22, 22]
    lonC = [41, 39, 40, 40, 38]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Huadong.pdf')

    #Huazhong
    latA = [79, 77, 62, 52, 56]
    lonA = [175, 169, 177, 172, 161]

    latG = [249, 248, 241, 236, 238]
    lonG = [587, 584, 588, 586, 580]

    latC = [20, 20, 16, 13, 14]
    lonC = [35, 34, 35, 34, 32]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw)) + 1

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Huazhong.pdf')

    #Huanan
    latA = [30, 32, 22, 30, 40]
    lonA = [152, 173, 161, 176, 160]

    latG = [225, 226, 221, 225, 230]
    lonG = [576, 586, 580, 588, 580]

    latC = [7, 8, 5, 7, 10]
    lonC = [31, 35, 32, 35, 32]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw)) + 1

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Huanan.pdf')

    #Xinan
    latA = [57, 62, 45, 40, 59]
    lonA = [145, 136, 145, 129, 84]

    latG = [238, 241, 232, 230, 239]
    lonG = [592, 568, 572, 564, 542]

    latC = [15, 16, 12, 10, 15]
    lonC = [29, 27, 29, 26, 17]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Xinan.pdf')

    #Xibei
    latA = [77, 84, 85, 115, 90]
    lonA = [156, 135, 126, 70, 129]

    latG = [248, 252, 252, 267, 255]
    lonG = [578, 567, 563, 535, 564]

    latC = [20, 22, 22, 30, 23]
    lonC = [31, 27, 25, 14, 26]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMax_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigMax_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigMax_ann_his[20:50, 6, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMax_ccsm_rcp26[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMax_ccsm_rcp85[14:45, 6, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMax_ccsm_rcp26[64:95, 6, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMax_ccsm_rcp85[64:95, 6, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMax_ann_rcp26[14:45, 6, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMax_ann_rcp85[14:45, 6, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMax_ann_rcp26[64:95, 6, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMax_ann_rcp85[64:95, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(b) Tmax (July)',
           xl='Year', yl='Tmax (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmax-future-Xibei.pdf')


    #Jan
    #Huabei
    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    latA = [99, 96, 92, 91, 102]
    lonA = [185, 188, 177, 170, 165]

    latG = [259, 258, 256, 255, 261]
    lonG = [592, 594, 588, 585, 582]

    latC = [26, 25, 24, 24, 26]
    lonC = [37, 38, 35, 34, 33]

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Huabei.pdf')

    #Dongbei
    latA = [123, 114, 114, 106, 129]
    lonA = [226, 220, 205, 213, 216]

    latG = [271, 267, 267, 263, 274]
    lonG = [613, 610, 612, 606, 608]

    latC = [32, 30, 30, 28, 34]
    lonC = [45, 44, 45, 43, 43]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw)) + 1 

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Dongbei.pdf')

    #Huadong
    latA = [64, 68, 61, 84, 85]
    lonA = [205, 194, 200, 201, 188]

    latG = [242, 244, 240, 252, 252]
    lonG = [602, 597, 600, 600, 594]

    latC = [17, 17, 16, 22, 22]
    lonC = [41, 39, 40, 40, 38]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw)) 

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Huadong.pdf')

    #Huazhong
    latA = [79, 77, 62, 52, 56]
    lonA = [175, 169, 177, 172, 161]

    latG = [249, 248, 241, 236, 238]
    lonG = [587, 584, 588, 586, 580]

    latC = [20, 20, 16, 13, 14]
    lonC = [35, 34, 35, 34, 32]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw)) + 1

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Huazhong.pdf')

    #Huanan
    latA = [30, 32, 22, 30, 40]
    lonA = [152, 173, 161, 176, 160]

    latG = [225, 226, 221, 225, 230]
    lonG = [576, 586, 580, 588, 580]

    latC = [7, 8, 5, 7, 10]
    lonC = [31, 35, 32, 35, 32]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Huanan.pdf')

    #Xinan
    latA = [57, 62, 45, 40, 59]
    lonA = [145, 136, 145, 129, 84]

    latG = [238, 241, 232, 230, 239]
    lonG = [592, 568, 572, 564, 542]

    latC = [15, 16, 12, 10, 15]
    lonC = [29, 27, 29, 26, 17]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Xinan.pdf')

    #Xibei
    latA = [77, 84, 85, 115, 90]
    lonA = [156, 135, 126, 70, 129]

    latG = [248, 252, 252, 267, 255]
    lonG = [578, 567, 563, 535, 564]

    latC = [20, 22, 22, 30, 23]
    lonC = [31, 27, 25, 14, 26]

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')
    yCpr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_near_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_near_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yCpr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp26_long_raw = np.zeros((31, 5), dtype='float32')
    yApr_rcp85_long_raw = np.zeros((31, 5), dtype='float32')

    yCpr_rcp26_near = np.zeros((31), dtype='float32')
    yCpr_rcp85_near = np.zeros((31), dtype='float32')
    yApr_rcp26_near = np.zeros((31), dtype='float32')
    yApr_rcp85_near = np.zeros((31), dtype='float32')
    yCpr_rcp26_long = np.zeros((31), dtype='float32')
    yCpr_rcp85_long = np.zeros((31), dtype='float32')
    yApr_rcp26_long = np.zeros((31), dtype='float32')
    yApr_rcp85_long = np.zeros((31), dtype='float32')

    for i in range(0, 5):
        ytr[:, i] = bigMin_cru_his[20:50, 0, latG[i], lonG[i]]
        yCtr[:, i] = bigMin_ccsm_his[20:50, 0, latC[i], lonC[i]]
        yACtr[:, i] = bigMin_ann_his[20:50, 0, latA[i], lonA[i]]
        yCpr_rcp26_near_raw[:, i] = bigMin_ccsm_rcp26[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp85_near_raw[:, i] = bigMin_ccsm_rcp85[14:45, 0, latC[i], lonC[i]]
        yCpr_rcp26_long_raw[:, i] = bigMin_ccsm_rcp26[64:95, 0, latC[i], lonC[i]]
        yCpr_rcp85_long_raw[:, i] = bigMin_ccsm_rcp85[64:95, 0, latC[i], lonC[i]]
        yApr_rcp26_near_raw[:, i] = bigMin_ann_rcp26[14:45, 0, latA[i], lonA[i]]
        yApr_rcp85_near_raw[:, i] = bigMin_ann_rcp85[14:45, 0, latA[i], lonA[i]]
        yApr_rcp26_long_raw[:, i] = bigMin_ann_rcp26[64:95, 0, latA[i], lonA[i]]
        yApr_rcp85_long_raw[:, i] = bigMin_ann_rcp85[64:95, 0, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)
    yCpr_rcp26_near_raw = np.mean(yCpr_rcp26_near_raw, axis = 1)
    yCpr_rcp85_near_raw = np.mean(yCpr_rcp85_near_raw, axis = 1)
    yCpr_rcp26_long_raw = np.mean(yCpr_rcp26_long_raw, axis = 1)
    yCpr_rcp85_long_raw = np.mean(yCpr_rcp85_long_raw, axis = 1)
    yApr_rcp26_near_raw = np.mean(yApr_rcp26_near_raw, axis = 1)
    yApr_rcp85_near_raw = np.mean(yApr_rcp85_near_raw, axis = 1)
    yApr_rcp26_long_raw = np.mean(yApr_rcp26_long_raw, axis = 1)
    yApr_rcp85_long_raw = np.mean(yApr_rcp85_long_raw, axis = 1)

    yCpr_rcp26_near = smooth(yCpr_rcp26_near_raw, 31)
    yCpr_rcp85_near = smooth(yCpr_rcp85_near_raw, 31)
    yApr_rcp26_near = smooth(yApr_rcp26_near_raw, 31)
    yApr_rcp85_near = smooth(yApr_rcp85_near_raw, 31)

    yCpr_rcp26_long = smooth(yCpr_rcp26_long_raw, 31)
    yCpr_rcp85_long = smooth(yCpr_rcp85_long_raw, 31)
    yApr_rcp26_long = smooth(yApr_rcp26_long_raw, 31)
    yApr_rcp85_long = smooth(yApr_rcp85_long_raw, 31)

    x_array = np.arange(0, 30)
    x1_array = np.arange(30, 61)
    x2_array = np.arange(61, 92)

    minX = min(min(ytr), min(yCtr), min(yACtr), min(yCpr_rcp85_long_raw), min(yCpr_rcp85_long_raw), min(yApr_rcp26_long_raw), min(yApr_rcp85_long_raw))
    maxX = max(max(ytr), max(yCtr), max(yACtr), max(yCpr_rcp85_long_raw), max(yCpr_rcp85_long_raw), max(yApr_rcp26_long_raw), max(yApr_rcp85_long_raw))

    plot2d(19, x=[x_array, x_array, x_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x1_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array, x2_array], 
           y=[ytr, yCtr, yACtr, yCpr_rcp26_near_raw, yCpr_rcp85_near_raw, yApr_rcp26_near_raw, yApr_rcp85_near_raw, yCpr_rcp26_near, yCpr_rcp85_near, yApr_rcp26_near, yApr_rcp85_near, yCpr_rcp26_long_raw, yCpr_rcp85_long_raw, yApr_rcp26_long_raw, yApr_rcp85_long_raw, yCpr_rcp26_long, yCpr_rcp85_long, yApr_rcp26_long, yApr_rcp85_long],
           color=['r', 'k', 'b', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple', 'lightgrey', 'dimgrey', 'pink', 'plum', 'grey', 'k', 'hotpink', 'purple'], 
           linestyle=['-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '--', '--', '--', '--', '-', '-', '-', '-'], 
           lw=[2.5, 1.8, 2, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5, 1.3, 1.3, 1.3, 1.3, 2.5, 2.5, 2.5, 2.5],
           #lb=['CRU (CLIM)', 'CCSM (CLIM)', 'SD (CLIM)', 
           #    'CCSM RCP2.6 (Near-term)', 'CCSM RCP8.5 (Near-term)', 'SD RCP2.6 (Near-term)', 'SD RCP8.5 (Near-term)',
           #    'MA-CCSM RCP2.6 (Near-term)', 'MA-CCSM RCP8.5 (Near-term)', 'MA-SD RCP2.6 (Near-term)', 'MA-SD RCP8.5 (Near-term)', 
           #    'CCSM RCP2.6 (Long-term)', 'CCSM RCP8.5 (Long-term)', 'SD RCP2.6 (Long-term)', 'SD RCP8.5 (Long-term)',
           #    'MA-CCSM RCP2.6 (Long-term)', 'MA-CCSM RCP8.5 (Long-term)', 'MA-SD RCP2.6 (Long-term)', 'MA-SD RCP8.5 (Long-term)'],
           ti='(a) Tmin (January)',
           xl='Year', yl='Tmin (Celsius)',
           #xlim=[minX, maxX],
           ylim=[minX, maxX],
           figsize=(13, 6),
           fn='./figures_future/plot-tmin-future-Xibei.pdf')




