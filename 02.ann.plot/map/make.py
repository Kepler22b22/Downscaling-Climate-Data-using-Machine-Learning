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
    with open('../../02.ann.BiasCorrection/9-non/China_his_temp_postwork.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his, bigMin_ann_his = \
        data_dict['bigMax_ann_his'], data_dict['bigMin_ann_his']

    #with open('../../data/future/tr_tmax.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMax_ground_his = data_dict['bigMax_ground_tr']

    #with open('../../data/future/tr_tmin.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    #bigMin_ground_his = data_dict['bigMin_ground_tr']

    with open('../../../data/cru/cru_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_cru_his, bigMin_cru_his = data_dict['bigMax_all'], data_dict['bigMin_all']

    with open('../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']

    with open('../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his.values.reshape(56, 365, 42, 57), bigMin_ccsm_his.values.reshape(56, 365, 42, 57)

    bigMax_ann_his, bigMin_ann_his = \
        monthlymean(bigMax_ann_his), monthlymean(bigMin_ann_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ccsm_his), monthlymean(bigMin_ccsm_his)

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his.values.reshape(56, 12, 360, 720), bigMin_cru_his.values.reshape(56, 12, 360, 720)

    bigMax_cru_his, bigMin_cru_his = \
        bigMax_cru_his[20:50, :, :, :], bigMin_cru_his[20:50, :, :, :]

    print('Data Loaded. ')

    #Mask
    #fCpr =  xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_tasmax_1948-2005.nc')
    #fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    #Lat, Lon = fCpr.lat, fCpr.lon
    #for j in range(42):
    #    for i in range(57):
    #        cur_lat, cur_lon = Lat[j], Lon[i]
    #        c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
    #        cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
    #        latc, lonc = int(cc_lat[0]), int(cc_lon[0])
    #        if m[latc, lonc]:
    #            continue
    #        else:
    #            bigMax_ccsm_his[:, :, j, i] = np.nan
    #            bigMin_ccsm_his[:, :, j, i] = np.nan
    
    #fCpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    #Latc, Lonc = fCpr.lat, fCpr.lon
    #for p in range(360):
    #    for q in range(720):
    #        cur_lat, cur_lon = Latc[p], Lonc[q]
    #        c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
    #        cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
    #        latc, lonc = int(cc_lat[0]), int(cc_lon[0])
    #        if m[latc, lonc]:
    #            continue
    #        else:
    #            bigMax_cru_his[:, :, p, q] = np.nan
    #            bigMin_cru_his[:, :, p, q] = np.nan

    #Plot map
    print('Now, doing plotting. ')

    #mon = July
    #yCtr = np.mean(bigMax_ccsm_his[20:50, 6, :, :], axis=0)
    #my.plot2d_map(yCtr - 273.15, Lon, Lat,
    #              levs = np.arange(-5, 45, 5),
    #              domain = [17, 54, 70, 140],
    #              cbar = 'jet', cbarstr = 'Tmax' +' (Celsius)',
    #              ti='CLIM(1970-1999) \nCCSM July', fout = './figures/subplot-tmax-ccsm.pdf' )

    #yGtr = np.mean(bigMax_cru_his[:, 6, :, :], axis=0)
    #my.plot2d_map(yGtr, Lonc, Latc,
    #              levs = np.arange(-5, 45, 5),
    #              domain = [17, 54, 70, 140],
    #              cbar = 'jet', cbarstr = 'Tmax' +' (Celsius)',
    #              ti='CLIM(1970-1999) \nCRU July', fout = './figures/subplot-tmax-cru.pdf' )

    yAtr = np.mean(bigMax_ann_his[20:50, 6, :, :], axis=0)
    my.plot2d_map(yAtr - 273.15, lon, lat,
                  levs = np.arange(-5, 45, 5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Tmax' +' (Celsius)',
                  ti='CLIM(1970-1999) \nSD July', fout = './figures/subplot-tmax-ann.pdf' )

    #mon = Jan
    yCtr = np.mean(bigMin_ccsm_his[20:50, 0, :, :], axis=0)
    my.plot2d_map(yCtr - 273.15, Lon, Lat,
                  levs = np.arange(-40, 20, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Tmin' +' (Celsius)',
                  ti='CLIM(1970-1999) \nCCSM Jan', fout = './figures/subplot-tmin-ccsm.pdf' )

    yGtr = np.mean(bigMin_cru_his[:, 0, :, :], axis=0)
    my.plot2d_map(yGtr, Lonc, Latc,
                  levs = np.arange(-40, 20, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Tmin' +' (Celsius)',
                  ti='CLIM(1970-1999) \nCRU Jan', fout = './figures/subplot-tmin-cru.pdf' )

    yAtr = np.mean(bigMin_ann_his[20:50, 0, :, :], axis=0)
    my.plot2d_map(yAtr - 273.15, lon, lat,
                  levs = np.arange(-40, 20, 5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'jet', cbarstr = 'Tmin' +' (Celsius)',
                  ti='CLIM(1970-1999) \nSD Jan', fout = './figures/subplot-tmin-ann.pdf' )









