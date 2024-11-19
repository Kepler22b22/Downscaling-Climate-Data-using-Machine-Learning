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
    with open('../01.bc.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his = data_dict['bigPr_bc_his']

    with open('../../data/ccsm/future/tr_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_his = data_dict['bigPr_ccsm_tr']
    
    with open('../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_cru_his = data_dict['bigPr_all']

    print(bigPr_cru_his.shape)
    print(bigPr_bc_his.shape)
    print(bigPr_ccsm_his.shape)

    bigPr_bc_his, bigPr_ccsm_his, bigPr_cru_his = \
        bigPr_bc_his.reshape(56, 365, 160, 280), \
        bigPr_ccsm_his.values.reshape(56, 365, 42, 57), \
        bigPr_cru_his.values.reshape(56, 12, 360, 720)

    bigPr_bc_his, bigPr_ccsm_his = \
        monthlymean(bigPr_bc_his) * 86400, \
        monthlymean(bigPr_ccsm_his) * 86400

    mm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(12):
        bigPr_cru_his[:, i] = bigPr_cru_his[:, i] / mm[i]

    print('Data Loaded. ')

    # Time series
    print('Now Doing Time Series...')
    
    # July
    #city = ['Harbin', 'Shenyang', 'Qsingdao', 'Zhengzhou', 'Shanghai', 
    #        'Guangzhou', 'Chengdu', 'Lasa', 'Urumuchi', 'Lanzhou', 'Beijing']

    region = ['Huabei', 'Dongbei', 'Huadong', 'Huazhong', 
              'Huanan', 'Xinan', 'Xibei']

    ytr = np.zeros((30, 5), dtype='float32')
    yCtr = np.zeros((30, 5), dtype='float32')
    yACtr = np.zeros((30, 5), dtype='float32')

    #Huabei
    latA = [99, 96, 92, 91, 102]
    lonA = [185, 188, 177, 170, 165]

    latG = [259, 258, 256, 255, 261]
    lonG = [592, 594, 588, 585, 582]

    latC = [26, 25, 24, 24, 26]
    lonC = [37, 38, 35, 34, 33]

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 2
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Huabei',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Huabei' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]
        print(ytr[:, i])
        print(yCtr[:, i])
        print(yACtr[:, i])

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Dongbei',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Dongbei' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 2
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Huadong',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Huadong' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Huazhong',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Huazhong' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Huanan',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Huanan' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Xinan',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Xinan' + '.pdf')

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

    for i in range(0, 5):
        ytr[:, i] = bigPr_cru_his[20:50, 6, latG[i], lonG[i]]
        yCtr[:, i] = bigPr_ccsm_his[20:50, 6, latC[i], lonC[i]]
        yACtr[:, i] = bigPr_bc_his[20:50, 6, latA[i], lonA[i]]

    ytr = np.mean(ytr, axis = 1)
    yCtr = np.mean(yCtr, axis = 1)
    yACtr = np.mean(yACtr, axis = 1)

    meanG = np.mean(ytr)
    meanC = np.mean(yCtr)
    meanA = np.mean(yACtr)

    stdG = np.std(ytr)
    stdC = np.std(yCtr)
    stdA = np.std(yACtr)

    x_array = np.arange(1970, 2000)

    minX = min(min(ytr), min(yCtr), min(yACtr)) - 1
    maxX = max(max(ytr), max(yCtr), max(yACtr))

    plot2d(3, x=[x_array, x_array, x_array], 
           y=[ytr, yCtr, yACtr],
           style=['r-o', 'k-o', 'b-o'], 
           lw=[2.5, 1.7, 2],
           lb=['CRU' + ' (' + '%.2f' % meanG + ', ' + '%.3f' % stdG + ')', 
               'CCSM' + ' (' + '%.2f' % meanC + ', '  +'%.3f' % stdC + ')', 
               'SD' + ' (' + '%.2f' % meanA + ', ' + '%.3f' % stdA + ')'],
           ti='(c) Pr (July)' + '\n' + 'Xibei',
           xl='Year', yl='Pr (mm/day)',
           xlim=[x_array[0], x_array[-1]],
           ylim=[minX, maxX, 1],
           figsize=(8, 4),
           fn='./figures/plot-prcp-China-historical-Xibei' + '.pdf')














