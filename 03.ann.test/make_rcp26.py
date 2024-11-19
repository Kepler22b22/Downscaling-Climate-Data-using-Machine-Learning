# MQ Zhang, Sep 2020

import os
import sys
import math
import pickle
import numpy             as np
import pandas            as pd
import xarray            as xr
import shelve            as sh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing  import StandardScaler
from sklearn.neural_network import MLPRegressor
from multiprocessing        import Process, Pool


def job_readdata_1(clon, clat):
    # Load datasets.
    fCpr = xr.open_dataset('../../data/ccsm4/train_vali/ccsm_tasmax_1948-2005.nc')

    # Find GCM's lat and lon.
    cur_lat, cur_lon = lat[clat], lon[clon]
    c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data_1(bigMax_ccsm, bigMin_ccsm):
        # Function to load and prepare data.
        dirs = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        xCmax_s, xCmin_s, xCpr_s = [], [], []

        # 1. Get C as INPUT
        for i in range(len(dirs)):
            cur_lon, cur_lat = lonc + dirs[i][0], latc + dirs[i][1]
            xCmax = bigMax_ccsm[:, cur_lat, cur_lon]
            xCmin = bigMin_ccsm[:, cur_lat, cur_lon]
            # Append
            xCmax, xCmin = np.array(xCmax), np.array(xCmin)
            xCmax_s.append(np.expand_dims(xCmax, axis=1))
            xCmin_s.append(np.expand_dims(xCmin, axis=1))
        xCmax_s, xCmin_s = np.concatenate(xCmax_s, axis=1), np.concatenate(xCmin_s, axis=1)

        # Prepare returned data.
        rnt_C = {'Tmax': xCmax_s, 'Tmin': xCmin_s}

        return rnt_C

    # Training
    xC = load_data_1(bigMax_ccsm_tr, bigMin_ccsm_tr)

    return xC


def job_readdata_2(clon, clat):
    # Load datasets.
    fCpr = xr.open_dataset('../../data/ccsm4/future/ccsm_pr_2006-2100_rcp26.nc')
    #fCmax = xr.open_dataset('../../data/ccsm4/future/ccsm_tasmax_2006-2100_rcp26.nc')
    #fCmin = xr.open_dataset('../../data/ccsm4/future/ccsm_tasmin_2006-2100_rcp26.nc')

    # Find GCM's lat and lon.
    cur_lat, cur_lon = lat[clat], lon[clon]
    c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data_2(bigMax_ccsm, bigMin_ccsm):
        # Function to load and prepare data.
        dirs = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        xCmax_s, xCmin_s, xCpr_s = [], [], []

        # 1. Get C as INPUT
        for i in range(len(dirs)):
            cur_lon, cur_lat = lonc + dirs[i][0], latc + dirs[i][1]
            xCmax = bigMax_ccsm[:, cur_lat, cur_lon]
            xCmin = bigMin_ccsm[:, cur_lat, cur_lon]
            # Append
            xCmax, xCmin = np.array(xCmax), np.array(xCmin)
            xCmax_s.append(np.expand_dims(xCmax, axis=1))
            xCmin_s.append(np.expand_dims(xCmin, axis=1))
        xCmax_s, xCmin_s = np.concatenate(xCmax_s, axis=1), np.concatenate(xCmin_s, axis=1)

        # Prepare returned data.
        rnt_C = {'Tmax': xCmax_s, 'Tmin': xCmin_s}

        return rnt_C

    # Training
    xCp = load_data_2(bigMax_ccsm_pr, bigMin_ccsm_pr)

    return xCp


def job_normalization(x1, x2):
    for key in x1.keys():
        scaler = StandardScaler()
        scaler.fit(x2[key])
        mean_ = scaler.mean_
        var_ = scaler.var_
        x1[key] = (x1[key] - mean_) / np.sqrt(var_)
    return x1


def job_concate(data_dict):
    buffer = [data_dict['Tmax'], data_dict['Tmin']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def job_ann_predict(ann, datain):
    return ann.predict(datain)


def info_year_day(y1, y2):
    d1 = (y1 - 2006) * 365
    d2 = (y2 + 1 - 2006) * 365 - 1
    Nyear = y2 - y1 + 1
    Nday = d2 - d1 + 1
    return (d1, d2, Nyear, Nday)


def job_save(name, j, i, lon, lat, xCp, xGp_ann):
    fname = "./out_temp_rcp26/ann_data_temp_rcp26." + name
    f = sh.open(fname)
    f['dataid'] = name
    f['arrayindex'] = (j, i)  # index in 0.25mask
    f['latlon'] = (lat, lon)
    f['data_predict'] = (xCp, xGp_ann)
    f.close()
    return fname


def workProcess(lat_range, m):
    for j in lat_range:
        for i in range(Nlon):
            if m[j, i]:
                dataid = "%05i" % (j * Nlon + i)        
                # 1. Read data
                Plon, Plat = lon[i], lat[j]
                xC = job_readdata_1(i, j)
                xCp = job_readdata_2(i, j) 
                #print('Read data Finished.')

                # 2. Normalization
                xCp = job_normalization(xCp, xC)
                #print('Normalization Finished.')
   
                # 3. Concatenate
                xCp = job_concate(xCp)
                #print('Concatenate Finished.')

                # 4. ANN
                fname = '../01.ann.train/out_temp/ann_data_temp.' + dataid
                f = sh.open(fname)
                myann = f['ann']
                f.close()
                xGp_ann = job_ann_predict(myann, xCp)
                #print('ANN Prediction Finished. ')

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

    # 2. Sin function
    #a = np.arange(365)
    #b = abs(a - 182)
    #c = b - 91
    #e = np.zeros(365)
    #for i in range(15 + 182):
    #    e[i] = c[i - 14 + 182]
    #for i in range(183 - 15):
    #    e[182 + 15 + i] = c[i]
    #e = e/91 / 2 * math.pi
    #for i in range(len(e)):
    #    e[i] = math.sin(e[i])

    # 3. Load Data
    with open('../../data/ccsm4/train_vali/ccsm.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_tr, bigMin_ccsm_tr = \
        data_dict['bigMax_ccsm_tr'], data_dict['bigMin_ccsm_tr']

    with open('../../data/ccsm4/future/ccsm_rcp_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_pr = data_dict['bigMax_ccsm_rcp26'] 

    with open('../../data/ccsm4/future/ccsm_rcp_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ccsm_pr = data_dict['bigMin_ccsm_rcp26']

    # 4. Multi_process
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
 
