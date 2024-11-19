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
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
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
    
    fpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    Latc, Lonc = fpr.lat, fpr.lon
    for p in range(360):
        for q in range(720):
            cur_lat, cur_lon = Latc[p], Lonc[q]
            c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[latc, lonc]:
                continue
            else:
                bigMax_cru_his[:, :, p, q] = np.nan
                bigMin_cru_his[:, :, p, q] = np.nan

    #Plot map
    print('Now, doing plotting. ')

    yGtr_max = np.mean(bigMax_cru_his[:, 6, :, :], axis=0)
    yGtr_min = np.mean(bigMin_cru_his[:, 0, :, :], axis=0)
    yatr_max = np.mean(bigMax_ann_his[20:50, 6, :, :], axis=0)
    yatr_min = np.mean(bigMin_ann_his[20:50, 0, :, :], axis=0)
    yctr_max = np.mean(bigMax_ccsm_his[20:50, 6, :, :], axis=0)
    yctr_min = np.mean(bigMin_ccsm_his[20:50, 0, :, :], axis=0)

    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    fCpr =  xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_tasmax_1948-2005.nc')
    fpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')

    Lat, Lon = fCpr.lat, fCpr.lon
    Latc, Lonc = fpr.lat, fpr.lon

    yAtr_max = np.zeros((Nlat, Nlon), dtype='float32')
    yAtr_min = np.zeros((Nlat, Nlon), dtype='float32')
    yCtr_max = np.zeros((42, 57), dtype='float32')
    yCtr_min = np.zeros((42, 57), dtype='float32')

    for j in range(Nlat):
        for i in range(Nlon):
            cur_lat, cur_lon = lat[j], lon[i]
            c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            C_lat, C_lon = fpr.lat.sel(lat=cur_lat, method='nearest'), fpr.lon.sel(lon=cur_lon, method='nearest')
            CC_lat, CC_lon = np.where(fpr.lat == C_lat), np.where(fpr.lon == C_lon)
            latC, lonC = int(CC_lat[0]), int(CC_lon[0])
            if m[j, i]:
                yAtr_max[j, i] = yatr_max[j, i] - yGtr_max[latC, lonC]
                yCtr_max[latc, lonc] = yctr_max[latc, lonc] - yGtr_max[latC, lonC]
                yAtr_min[j, i] = yatr_min[j, i] - yGtr_min[latC, lonC]
                yCtr_min[latc, lonc] = yctr_min[latc, lonc] - yGtr_min[latC, lonC]
            else:
                continue

    with open('./map_temp.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann': yAtr_max,
                     'bigMin_ann': yAtr_min, 
                     'bigMax_ccsm': yCtr_max,
                     'bigMin_ccsm': yCtr_min,
                     'bigMax_cru': yGtr_max,
                     'bigMin_cru': yGtr_min}, f, protocol=4)

    os.sys(exit())

    #mon = July
    my.plot2d_map(yCtr_max - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = 'Tmax' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmax-ccsm-2.pdf' )

    my.plot2d_map(yGtr_max, Lonc, Latc,
                  levs = np.arange(-5, 45, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = 'Tmax' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmax-cru.pdf' )

    my.plot2d_map(yAtr_max - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = 'Tmax' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmax-ann-2.pdf' )

    #mon = Jan
    my.plot2d_map(yCtr_min - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = 'Tmin' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmin-ccsm-2.pdf' )

    my.plot2d_map(yGtr_min, Lonc, Latc,
                  levs = np.arange(-40, 20, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = 'Tmin' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmin-cru.pdf' )

    my.plot2d_map(yAtr_min - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = 'Tmin' +' (Celsius)',
                  ti='', fout = './figures/subplot-tmin-ann-2.pdf' )









