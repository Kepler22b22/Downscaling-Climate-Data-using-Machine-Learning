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

    with open('../../../data/ccsm4/future/tr_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_his = data_dict['bigPr_ccsm_tr']

    with open('../../03.bc.test/China_bc_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_bc_rcp26, bigPr_bc_rcp85 = data_dict['bigPr_pr_rcp26'], data_dict['bigPr_pr_rcp85']

    with open('../../../data/ccsm4/future/ccsm_rcp_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_rcp26, bigPr_ccsm_rcp85 = \
        data_dict['bigPr_ccsm_rcp26'], data_dict['bigPr_ccsm_rcp85']

    bigPr_bc_his, bigPr_ccsm_his = \
        bigPr_bc_his * 86400, bigPr_ccsm_his * 86400

    bigPr_bc_his = monthlymean(bigPr_bc_his)

    bigPr_ccsm_his = bigPr_ccsm_his.values.reshape(56, 365, 42, 57)
    bigPr_ccsm_his = monthlymean(bigPr_ccsm_his)
 
    bigPr_bc_rcp26, bigPr_ccsm_rcp26 = \
        bigPr_bc_rcp26*86400, bigPr_ccsm_rcp26*86400
    
    bigPr_bc_rcp26 = bigPr_bc_rcp26.reshape(95, 365, 160, 280)
    bigPr_bc_rcp85 = bigPr_bc_rcp85.reshape(95, 365, 160, 280)
    bigPr_ccsm_rcp26 = bigPr_ccsm_rcp26.values.reshape(95, 365, 42, 57)
    bigPr_ccsm_rcp85 = bigPr_ccsm_rcp85.values.reshape(95, 365, 42, 57)

    bigPr_bc_rcp26 = monthlymean(bigPr_bc_rcp26)
    bigPr_ccsm_rcp26 = monthlymean(bigPr_ccsm_rcp26)

    bigPr_bc_rcp85, bigPr_ccsm_rcp85 = \
        bigPr_bc_rcp85*86400, bigPr_ccsm_rcp85*86400
    
    bigPr_bc_rcp85 = monthlymean(bigPr_bc_rcp85)
    bigPr_ccsm_rcp85 = monthlymean(bigPr_ccsm_rcp85)

    print('Data Loaded. ')

    #Mask
    fCpr =  xr.open_dataset('../../../data/ccsm4/future/ccsm_pr_2006-2100_rcp26.nc')
    fGpr = xr.open_dataset('../../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')
    Lat, Lon = fCpr.lat, fCpr.lon
    
    yAtr = np.mean(bigPr_bc_his[20:50, 6, :, :], axis=0)
    yCtr = np.mean(bigPr_ccsm_his[20:50, 6, :, :], axis=0)
    yapr_rcp26_near = np.mean(bigPr_bc_rcp26[14:44, 6, :, :], axis=0)
    yapr_rcp26_long = np.mean(bigPr_bc_rcp26[64:94, 6, :, :], axis=0)
    ycpr_rcp26_near = np.mean(bigPr_ccsm_rcp26[14:44, 6, :, :], axis=0)
    ycpr_rcp26_long = np.mean(bigPr_ccsm_rcp26[64:94, 6, :, :], axis=0)
    yapr_rcp85_near = np.mean(bigPr_bc_rcp85[14:44, 6, :, :], axis=0)
    yapr_rcp85_long = np.mean(bigPr_bc_rcp85[64:94, 6, :, :], axis=0)
    ycpr_rcp85_near = np.mean(bigPr_ccsm_rcp85[14:44, 6, :, :], axis=0)
    ycpr_rcp85_long = np.mean(bigPr_ccsm_rcp85[64:94, 6, :, :], axis=0)

    yApr_rcp26_near = np.zeros((Nlat, Nlon), dtype='float32')
    yApr_rcp26_long = np.zeros((Nlat, Nlon), dtype='float32')
    yCpr_rcp26_near = np.zeros((42, 57), dtype='float32')
    yCpr_rcp26_long = np.zeros((42, 57), dtype='float32')
    yApr_rcp85_near = np.zeros((Nlat, Nlon), dtype='float32')
    yApr_rcp85_long = np.zeros((Nlat, Nlon), dtype='float32')
    yCpr_rcp85_near = np.zeros((42, 57), dtype='float32')
    yCpr_rcp85_long = np.zeros((42, 57), dtype='float32')

    for j in range(Nlat):
        for i in range(Nlon):
            cur_lat, cur_lon = lat[j], lon[i]
            c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
            cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
            latc, lonc = int(cc_lat[0]), int(cc_lon[0])
            if m[j, i]:
                yApr_rcp26_near[j, i] = yapr_rcp26_near[j, i] - yAtr[j, i]
                yApr_rcp26_long[j, i] = yapr_rcp26_long[j, i] - yAtr[j, i]
                yCpr_rcp26_near[latc, lonc] = ycpr_rcp26_near[latc, lonc] - yCtr[latc, lonc]
                yCpr_rcp26_long[latc, lonc] = ycpr_rcp26_long[latc, lonc] - yCtr[latc, lonc]
                yApr_rcp85_near[j, i] = yapr_rcp85_near[j, i] - yAtr[j, i]
                yApr_rcp85_long[j, i] = yapr_rcp85_long[j, i] - yAtr[j, i]
                yCpr_rcp85_near[latc, lonc] = ycpr_rcp85_near[latc, lonc] - yCtr[latc, lonc]
                yCpr_rcp85_long[latc, lonc] = ycpr_rcp85_long[latc, lonc] - yCtr[latc, lonc]
            else:
                yApr_rcp26_near[j, i] = np.nan
                yApr_rcp26_long[j, i] = np.nan
                yCpr_rcp26_near[latc, lonc] = np.nan
                yCpr_rcp26_long[latc, lonc] = np.nan
                yApr_rcp85_near[j, i] = np.nan
                yApr_rcp85_long[j, i] = np.nan
                yCpr_rcp85_near[latc, lonc] = np.nan
                yCpr_rcp85_long[latc, lonc] = np.nan
                
    #Plot map
    print('Now, doing plotting. ')

    #mon = July, Near-term
    my.plot2d_map(yCpr_rcp26_near, Lon, Lat,
                  levs = np.arange(-4.5, 5.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-ccsm-rcp26-1.pdf' )
    
    my.plot2d_map(yApr_rcp26_near, lon, lat,
                  levs = np.arange(-4.5, 5.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-bc-rcp26-1.pdf' )

    my.plot2d_map(yCpr_rcp85_near, Lon, Lat,
                  levs = np.arange(-3.5, 4.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-ccsm-rcp85-1.pdf' )
    
    my.plot2d_map(yApr_rcp85_near, lon, lat,
                  levs = np.arange(-3.5, 4.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-bc-rcp85-1.pdf' )

    #mon = July, Long-term
    my.plot2d_map(yCpr_rcp26_long, Lon, Lat,
                  levs = np.arange(-3.5, 4.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-ccsm-rcp26-2.pdf' )
    
    my.plot2d_map(yApr_rcp26_long, lon, lat,
                  levs = np.arange(-3.5, 4.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-bc-rcp26-2.pdf' )

    my.plot2d_map(yCpr_rcp85_long, Lon, Lat,
                  levs = np.arange(-4.5, 5.5, 0.5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-ccsm-rcp85-2.pdf' )
    
    my.plot2d_map(yApr_rcp85_long, lon, lat,
                  levs = np.arange(-4.5, 5.5, 0.5), 
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-pr-bc-rcp85-2.pdf' )




