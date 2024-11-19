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
    bigMax_linear_tr = np.zeros((15330, Nlat, Nlon), dtype='float32')
    bigMin_linear_tr = np.zeros((15330, Nlat, Nlon), dtype='float32')
    bigMax_linear_pr = np.zeros((5110, Nlat, Nlon), dtype='float32')
    bigMin_linear_pr = np.zeros((5110, Nlat, Nlon), dtype='float32')
    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)
                print("%s: In China！！j=%i, i=%i" % (dataid, j, i))
                fname = "./out_temp/lr_data_temp." + dataid
                with sh.open(fname) as f:
                    # 1. read data from the tuple:(xCp, xGp, xGp_linear)
                    data_linear_pr = f['data_predict'][2]
                    data_linear_tr = f['data_train'][2]
                # 2. put into Big array
                bigMax_linear_tr[:, j, i] = data_linear_tr[:, 0]
                bigMax_linear_pr[:, j, i] = data_linear_pr[:, 0]
                bigMin_linear_tr[:, j, i] = data_linear_tr[:, 1]
                bigMin_linear_pr[:, j, i] = data_linear_pr[:, 1]

    return bigMax_linear_tr, bigMax_linear_pr, bigMin_linear_tr, bigMin_linear_pr


def monthlymean(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    nd_ = bigArray.size
    ny_ = nd_ // 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


def plot2d(N, x, y, style, lw, lb,
           ti='Plot', xl='X', yl='Y', legendloc=4,
           xlim=[], ylim=[], ylog=False,
           figsize=(10, 6),
           fn="plot2d.pdf",
           cell_text=[], rows=[],
           columns=[]):
    plt.figure(figsize=figsize)
    
    rowcolors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colcolors = plt.cm.BuPu(np.linspace(0, 0.5, len(columns)))

    # reverse colors and text labels to display the last value at the top
    rowcolors = rowcolors[::-1]
    colcolors = colcolors[::-1]
    cell_text.reverse()

    # add a table at the bottom of the figur
    the_table = plt.table(cellText = cell_text, cellLoc = 'center', colWidths=[0.1]*3,
                          rowLabels = rows, rowColours = rowcolors, 
                          colLabels = columns, colColours = colcolors,  loc='best',
                          bbox = [0.8, 0.8, 0.21, 0.2])

    # adjust layout to make room for the table
    plt.subplots_adjust(left=0.2, bottom=0.4)

    for i in range(N):
        plt.plot(x[i], y[i], style[i], linewidth=lw[i], label=lb[i], alpha=0.55)

    plt.title(ti)
    plt.xlabel(xl)
    plt.ylabel(yl)

    if xlim: plt.xlim(xlim[0], xlim[1])
    if ylim: plt.ylim(ylim[0], ylim[1])
    if ylog: plt.yscale('log')

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

    # Combine
    bigMax_linear_tr, bigMax_linear_pr, bigMin_linear_tr, bigMin_linear_pr = load_data()

    # Dump and Load
    with open('China_linear_temp.pkl', 'wb') as f:
        pickle.dump({'bigMax_linear_tr': bigMax_linear_tr,                 
                     'bigMax_linear_pr': bigMax_linear_pr, 
                     'bigMin_linear_tr': bigMin_linear_tr, 
                     'bigMin_linear_pr': bigMin_linear_pr}, f)





