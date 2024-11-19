# Zhang MQ, July 2020

import os
import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy      as np
import xarray     as xr
import shelve     as sh
import plotfunc   as my


def monthlymean(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    ny_, nd_, nlat_, nlon_ = bigArray.shape
    #ny_ = nd_// 365
    meanArray = np.zeros((ny_, 12, nlat_, nlon_), dtype='float32')
    #bigArray = bigArray.reshape(ny_, 365, nlat_, nlon_)
    for i in range(12):
        meanArray[:, i, :, :] = np.mean(bigArray[:, m_aa[i]:m_bb[i], :, :], axis=1)
    return meanArray


if __name__=='__main__':
    # MAIN LOOP for each grid in China
    fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    #Load data
    with open('../../01.bc.train/China_his_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_his = data_dict['bigPr_bc_his']

    with open('../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_cru_his = data_dict['bigPr_all']

    with open('../../../data/ccsm4/future/tr_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_his = data_dict['bigPr_ccsm_tr']

    bigPr_bc_his, bigPr_ccsm_his = \
        bigPr_bc_his * 86400, bigPr_ccsm_his * 86400

    bigPr_bc_his = monthlymean(bigPr_bc_his)

    bigPr_ccsm_his = bigPr_ccsm_his.values.reshape(56, 365, 42, 57)
    bigPr_ccsm_his = monthlymean(bigPr_ccsm_his)

    bigPr_cru_his = bigPr_cru_his.values.reshape(56, 12, 360, 720)
    bigPr_cru_his = bigPr_cru_his[20:50, :, :, :]

    mm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(12):
        bigPr_cru_his[:, i, :, :] = bigPr_cru_his[:, i, :, :] / mm[i]

    print('Data Loaded. ')

    #Mask
    fCpr =  xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_pr_1948-2005.nc')
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    Lat, Lon = fCpr.lat, fCpr.lon
    for j in range(42):
        for i in range(57):
            cur_lat, cur_lon = Lat[j], Lon[i]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                continue
            else:
                bigPr_ccsm_his[:, :, j, i] = np.nan
    
    fCpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    Latc, Lonc = fCpr.lat, fCpr.lon
    for p in range(360):
        for q in range(720):
            cur_lat, cur_lon = Latc[p], Lonc[q]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                continue
            else:
                bigPr_cru_his[:, :, p, q] = np.nan

    for j in range(Nlat):
        for i in range(Nlon):
            if m[j, i]:
                continue
            else:
                bigPr_bc_his[:, :, j, i] = np.nan

    #Plot map
    print('Now, doing plotting. ')

    #mon = July
    yAtr = np.mean(bigPr_bc_his[20:50, 6, :, :], axis=0)
    my.plot2d_map(yCtr, Lon, Lat,
                  levs = np.arange(0, 20, 2),
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Pr' +' (mm/day)',
                  ti='CLIM(1970-1999) \nCCSM July', fout = './figures/subplot-pr-ccsm.pdf' )
    
    yGtr = np.mean(bigPr_cru_his[:, 6, :, :], axis=0) 
    my.plot2d_map(yGtr, Lonc, Latc,
                  levs = np.arange(0, 20, 2),
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Pr' +' (mm/day)',
                  ti='CLIM(1970-1999) \nCRU July', fout = './figures/subplot-pr-cru.pdf' )

    yAtr = np.mean(bigPr_bc_his[20:50, 6, :, :], axis=0)
    my.plot2d_map(yAtr, lon, lat,
                  levs = np.arange(0, 20, 2), 
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Pr' +' (mm/day)',
                  ti='CLIM(1970-1999) \nSD July', fout = './figures/subplot-pr-bc.pdf' )




