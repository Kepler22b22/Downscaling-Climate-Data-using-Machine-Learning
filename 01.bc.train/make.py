# MQ Zhang, Sep 2019

import os
import sys
import pickle
import numpy             as np
import pandas            as pd
import xarray            as xr
import matplotlib.pyplot as plt
import shelve            as sh

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LinearRegression
from multiprocessing       import Process, Pool
from sklearn.metrics       import mean_squared_error
from sklearn.metrics       import r2_score

def job_readdata(clon, clat):
    # Load datasets.
    path = '../../data/ccsm4/train_vali/'
    fCpr = xr.open_dataset(path + 'ccsm_pr_1948-2005.nc')
    fGpr = xr.open_dataset(path + 'gmfd_prcp_1948-2005.nc')

    # Find GCM's lat and lon.
    cur_lat, cur_lon = lat[clat], lon[clon]
    c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data(bigPr_ccsm, bigPr_ground):
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

        # 2. get G as OUTPUT
        cur_lon, cur_lat = lon[clon], lat[clat]
        g_lat, g_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
        gg_lat, gg_lon = np.where(fGpr.lat == g_lat), np.where(fGpr.lon == g_lon)
        latG, lonG = int(gg_lat[0]), int(gg_lon[0])
        xGpr = bigPr_ground[:, latG, lonG]

        # Convert Data Format to np.array with Shape (sample_num, 1).
        xGpr = np.array(xGpr)
        xGpr = np.expand_dims(xGpr, axis=1)

        # scale the value of pr
        #scale = 1e7
        #xCpr_s *= scale
        #xGpr *= scale

        # Prepare returned data.
        rnt_C = {'Pr': xCpr_s}
        rnt_G = {'Pr': xGpr}

        return rnt_C, rnt_G

    # Training
    xC, xG = load_data(bigPr_ccsm_tr, bigPr_ground_tr)

    # Prediction
    xCp, xGp = load_data(bigPr_ccsm_pr, bigPr_ground_pr)

    return (xC, xG, xCp, xGp)


def job_concate(data_dict):
    buffer = [data_dict['Pr']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def get_newshape(bigArray):
    nd_, none = bigArray.shape
    ny_ = nd_ // 365
    bigArray = bigArray.reshape(ny_, 365)
    return bigArray


def get_alpha(bigPr_ccsm, bigPr_ground):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    ny_, none = bigPr_ccsm.shape
    alpha_new = np.zeros((12))
    for i in range(12):
        big_ccsm = bigPr_ccsm[:,m_aa[i]:m_bb[i]].reshape(-1)
        big_ground = bigPr_ground[:,m_aa[i]:m_bb[i]].reshape(-1)
        alpha = np.sum(big_ground) / np.sum(big_ccsm)
        alpha_new[i] = alpha
    return alpha_new


def predict(alpha, bigPr_ccsm):
    ny_, none = bigPr_ccsm.shape
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    bigPr_amplified = np.zeros((ny_, 365), dtype='float32')
    for i in range(12):
        bigPr_amplified[:, m_aa[i]:m_bb[i]] = bigPr_ccsm[:, m_aa[i]:m_bb[i]] * alpha[i]
    bigPr_amplified = bigPr_amplified.reshape(-1)
    return bigPr_amplified


def job_save(name, j, i, lon, lat, xC, xG, xG_amplified, xCp, xGp, bigPr_amplified, alpha):
    fname = "./out/bc_data." + name
    f = sh.open(fname)
    f['dataid'] = name
    f['arrayindex'] = (j, i)  # index in 0.25mask
    f['latlon'] = (lat, lon)
    f['data_train'] = (xC, xG, xG_amplified)
    f['data_predict'] = (xCp, xGp, bigPr_amplified)
    f['alpha'] = alpha
    f.close()
    return fname


def workProcess(lat_range, m):
    for j in lat_range:
        for i in range(Nlon):
            if m[j, i]:
                # 1. Read data
                Plon, Plat = lon[i], lat[j]
                xC, xG, xCp, xGp = job_readdata(i, j)
                # print('Read data Finished.')

                # 2. Concatenate
                xC, xG, xCp, xGp = job_concate(xC), job_concate(xG), job_concate(xCp), job_concate(xGp)
                # print('Concatenate Finished.')

                # 3. get_newshape
                xC, xCp, xG, xGp = get_newshape(xC), get_newshape(xCp), get_newshape(xG), get_newshape(xGp)
                # print('Get new shape Finished.')

                # 4. get_alpha
                alpha = get_alpha(xC, xG)
                #print(alpha)
                # print('Get Alpha Finished.')

                # 5. Use Alpha
                xG_amplified = predict(alpha, xC)
                bigPr_amplified = predict(alpha, xCp)
                # print('Use Alpha Finished. ')

                # 6. Log
                mse = mean_squared_error(xGp.reshape(-1), bigPr_amplified.reshape(-1))
                rmse = np.sqrt(mse)
                cc = np.corrcoef(xGp.reshape(-1), bigPr_amplified.reshape(-1))
                print('rmse = %3.3f' % rmse)
                print('cc = %3.3f' % cc[0][1])

                # 7. Save
                dataid = "%05i" % (j * Nlon + i)
                job_save(dataid, j, i, Plon, Plat, xC, xG, xG_amplified, xCp, xGp, bigPr_amplified, alpha)
                # print('Save Data Finished.')
    return



if __name__ == '__main__':
    # Basic parameters
    # 1. Mask
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    #2. Load datasets
    with open('../../data/ccsm4/train_vali/ccsm.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ccsm_tr, bigPr_ccsm_pr = \
      data_dict['bigPr_ccsm_tr'], data_dict['bigPr_ccsm_pr']

    with open('../../data/ccsm4/train_vali/gmfd_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ground_tr, bigPr_ground_pr = \
      data_dict['bigPr_ground_tr'], data_dict['bigPr_ground_pr']

    # Multi_process
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

