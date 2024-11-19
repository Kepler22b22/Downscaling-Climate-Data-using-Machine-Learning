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

    with open('../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_his = data_dict['bigMax_ccsm_tr']

    with open('../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_his = data_dict['bigMin_ccsm_tr']

    with open('../../../data/ccsm4/future/ccsm_rcp_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_rcp26, bigMax_ccsm_rcp85 = \
        data_dict['bigMax_ccsm_rcp26'], data_dict['bigMax_ccsm_rcp85']

    with open('../../../data/ccsm4/future/ccsm_rcp_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_rcp26, bigMin_ccsm_rcp85 = \
        data_dict['bigMin_ccsm_rcp26'], data_dict['bigMin_ccsm_rcp85']

    with open('../../04.ann.BiasCorrection/China_tmax_post_rcp85.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_rcp85 = data_dict['bigMax_ann_rcp85']

    with open('../../04.ann.BiasCorrection/China_tmin_post_rcp85.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ann_rcp85 = data_dict['bigMin_ann_rcp85']
    
    bigMax_ccsm_his, bigMin_ccsm_his = \
        bigMax_ccsm_his.values.reshape(56, 365, 42, 57), bigMin_ccsm_his.values.reshape(56, 365, 42, 57)

    bigMax_ann_his, bigMin_ann_his = \
        monthlymean(bigMax_ann_his), monthlymean(bigMin_ann_his)

    bigMax_ccsm_his, bigMin_ccsm_his = \
        monthlymean(bigMax_ccsm_his), monthlymean(bigMin_ccsm_his)    
    
    bigMax_ann_rcp85 = np.squeeze(np.array(bigMax_ann_rcp85))
    bigMin_ann_rcp85 = np.squeeze(np.array(bigMin_ann_rcp85))
    
    bigMax_ccsm_rcp85, bigMin_ccsm_rcp85 = \
        bigMax_ccsm_rcp85.values.reshape(95, 365, 42, 57), \
        bigMin_ccsm_rcp85.values.reshape(95, 365, 42, 57)

    bigMax_ann_rcp85, bigMin_ann_rcp85 = \
        monthlymean(bigMax_ann_rcp85), monthlymean(bigMin_ann_rcp85)

    bigMax_ccsm_rcp85, bigMin_ccsm_rcp85 = \
        monthlymean(bigMax_ccsm_rcp85), monthlymean(bigMin_ccsm_rcp85)

    print('Data Loaded. ')

    #Mask
    fpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    fCpr =  xr.open_dataset('../../../data/ccsm4/future/ccsm_pr_2006-2100_rcp26.nc')
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    Lat, Lon = fCpr.lat, fCpr.lon
    #for j in range(42):
    #    for i in range(57):
    #        cur_lat, cur_lon = Lat[j], Lon[i]
    #        c_lat, c_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
    #        cc_lat, cc_lon =  np.where(fGpr.lat == c_lat), np.where(fGpr.lon == c_lon)
    #        latc, lonc = int(cc_lat[0]), int(cc_lon[0])
    #        if m[latc, lonc]:
    #            continue
    #        else:
    #            bigMax_ccsm_rcp26[:, :, j, i] = np.nan
    #            bigMin_ccsm_rcp26[:, :, j, i] = np.nan
    #            bigMax_ccsm_rcp85[:, :, j, i] = np.nan
    #            bigMin_ccsm_rcp85[:, :, j, i] = np.nan
    
    yAtr_max = np.mean(bigMax_ann_his[20:50, 6, :, :], axis=0)
    yAtr_min = np.mean(bigMin_ann_his[20:50, 0, :, :], axis=0)
    yCtr_max = np.mean(bigMax_ccsm_his[20:50, 6, :, :], axis=0)
    yCtr_min = np.mean(bigMin_ccsm_his[20:50, 0, :, :], axis=0)
    yapr_max = np.mean(bigMax_ann_rcp85[14:44, 6, :, :], axis=0)
    yapr_min = np.mean(bigMin_ann_rcp85[14:44, 0, :, :], axis=0)
    ycpr_max = np.mean(bigMax_ccsm_rcp85[14:44, 6, :, :], axis=0)
    ycpr_min = np.mean(bigMin_ccsm_rcp85[14:44, 0, :, :], axis=0)

    yApr_max = np.zeros((Nlat, Nlon), dtype='float32')
    yApr_min = np.zeros((Nlat, Nlon), dtype='float32')
    yCpr_max = np.zeros((42, 57), dtype='float32')
    yCpr_min = np.zeros((42, 57), dtype='float32')

    for j in range(Nlat):
        for i in range(Nlon):
            cur_lat, cur_lon = lat[j], lon[i]
            c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[j, i]:
                yApr_max[j, i] = yapr_max[j, i] - yAtr_max[j, i]
                yCpr_max[latc, lonc] = ycpr_max[latc, lonc] - yCtr_max[latc, lonc]
                yApr_min[j, i] = yapr_min[j, i] - yAtr_min[j, i]
                yCpr_min[latc, lonc] = ycpr_min[latc, lonc] - yCtr_min[latc, lonc]
            else:
                yApr_max[j, i] = np.nan
                yCpr_max[latc, lonc] = np.nan
                yApr_min[j, i] = np.nan
                yCpr_min[latc, lonc] = np.nan

    #Plot map
    print('Now, doing plotting. ')

    #mon = July, Long-term
    my.plot2d_map(yCpr_max, Lon, Lat,
                  levs = np.arange(-7.5, 7.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ccsm-rcp85-1.pdf' )

    my.plot2d_map(yApr_max, lon, lat,
                  levs = np.arange(-7.5, 7.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ann-rcp85-1.pdf' )

    #mon = Jan, Long-term
    my.plot2d_map(yCpr_min, Lon, Lat,
                  levs = np.arange(-7.5, 7.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ccsm-rcp85-1.pdf' )

    my.plot2d_map(yApr_min, lon, lat,
                  levs = np.arange(-7.5, 7.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ann-rcp85-1.pdf' )









