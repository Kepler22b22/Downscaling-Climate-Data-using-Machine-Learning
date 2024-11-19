# MQ Zhang, Sep 2020

import os
import sys
import math
import pickle
import numpy             as np
import pandas            as pd
import xarray            as xr
import shelve            as sh
import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings('ignore')

from sklearn.preprocessing  import StandardScaler
from sklearn.neural_network import MLPRegressor
from multiprocessing        import Process, Pool


def job_readdata(clon, clat):
    # Load datasets.
    fCpr = xr.open_dataset('../../data/ccsm4/future/ccsm_pr_2006-2100_rcp85.nc')

    # Find GCM's lat and lon.
    cur_lat, cur_lon = lat[clat], lon[clon]
    c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data(bigPr_ccsm):
        # Function to load and prepare data.
        dirs = [[0, 0]]
        xCmax_s, xCmin_s, xCpr_s = [], [], []

        # 1. Get C as INPUT
        for i in range(len(dirs)):
            cur_lon, cur_lat = lonc + dirs[i][0], latc + dirs[i][1]
            xCpr = bigPr_ccsm[:, cur_lat, cur_lon]
            # Append
            xCpr = np.array(xCpr)
            xCpr_s.append(np.expand_dims(xCpr, axis=1))
        xCpr_s = np.concatenate(xCpr_s, axis=1)

        # Prepare returned data.
        rnt_C = {'Pr': xCpr_s}

        return rnt_C

    # Training
    xCp = load_data(bigPr_ccsm_pr)

    return xCp


def job_concate(data_dict):
    buffer = [data_dict['Pr']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def job_bc_predict(alpha, datain):
    nd_, none = datain.shape
    ny_ = nd_ // 365
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    datain = datain.reshape(ny_, 365)
    bigPr_amplified = np.zeros((ny_, 365), dtype='float32')
    for i in range(12):
        bigPr_amplified[:, m_aa[i]:m_bb[i]] = datain[:, m_aa[i]:m_bb[i]] * alpha[i]
    bigPr_amplified = bigPr_amplified.reshape(-1)
    return bigPr_amplified


def info_year_day(y1, y2):
    d1 = (y1 - 2006) * 365
    d2 = (y2 + 1 - 2006) * 365 - 1
    Nyear = y2 - y1 + 1
    Nday = d2 - d1 + 1
    return (d1, d2, Nyear, Nday)


def job_save(name, j, i, lon, lat, xCp, xGp_bc):
    fname = "./out_pr_rcp85/bc_data_pr_rcp85." + name
    f = sh.open(fname)
    f['dataid'] = name
    f['arrayindex'] = (j, i)  # index in 0.25mask
    f['latlon'] = (lat, lon)
    f['data_predict'] = (xCp, xGp_bc)
    f.close()
    return fname


def workProcess(lat_range, m):
    for j in lat_range:
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)        
                # 1. Read data
                Plon, Plat = lon[i], lat[j]
                xCp = job_readdata(i, j) 
                # print('Read data Finished.')

                # 2. Concatenate
                xCp = job_concate(xCp)
                # print('Concatenate Finished.')
   
                # 3. BC
                fname = '../01.bc.train/out/bc_data.' + dataid
                f = sh.open(fname)
                alpha = f['alpha']
                f.close()
                xGp_ann = job_bc_predict(alpha, xCp)
                # print('BC Prediction Finished. ')

                # 5. Log
                print('Now: j=%3d, i=%3d ' % (j, i))

                # 6. Save
                job_save(dataid, j, i, Plon, Plat, xCp, xGp_ann)
                # print('Save Data Finished.')
    return


if __name__ == '__main__':
    # 1. Mask
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # 2. Load Data
    with open('../../data/ccsm4/future/ccsm_rcp_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_pr = data_dict['bigPr_ccsm_rcp85'] 

    # 3. Multi_process
    process_num = 1
    chunck_size = Nlat // process_num
    lat_ranges = [range(id, min(Nlat, id + chunck_size)) for id in range(0, Nlat, chunck_size)]
    p = Pool(process_num)
    for lat_range in lat_ranges:
        print(lat_range)
        p.apply_async(workProcess, args=(lat_range, m))
    p.close()
    p.join()
    print('End of main thread...')
    #workProcess(None, m)
 
