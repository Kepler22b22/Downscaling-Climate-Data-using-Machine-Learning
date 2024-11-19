# MQ Zhang, Sep 2019

import os
import sys
import math
import pickle
import numpy             as np
import pandas            as pd
import xarray            as xr
import shelve            as sh
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing  import StandardScaler
from sklearn.neural_network import MLPRegressor
from multiprocessing        import Process, Pool


def job_readdata(clon, clat):
    # Load datasets.
    fCpr = xr.open_dataset('../../data/ccsm4/train_vali/ccsm_pr_1948-2005.nc')
    fGpr = xr.open_dataset('../../data/ccsm4/train_vali/gmfd_prcp_1948-2005.nc')

    # Find GCM's lat and lon.
    cur_lat, cur_lon = lat[clat], lon[clon]
    c_lat, c_lon = fCpr.lat.sel(lat=cur_lat, method='nearest'), fCpr.lon.sel(lon=cur_lon, method='nearest')
    print('c_lat = ', c_lat)
    print('c_lon = ', c_lon)
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    print(cc_lat)
    print(cc_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data(bigMax_ccsm, bigMin_ccsm, bigPr_ccsm, bigPr_ground):
        # Function to load and prepare data.
        #dirs = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        dirs = [[0, 0]]
        xCmax_s, xCmin_s, xCpr_s = [], [], []

        # 1. Get C as INPUT
        for i in range(len(dirs)):
            cur_lon, cur_lat = lonc + dirs[i][0], latc + dirs[i][1]
            xCmax = bigMax_ccsm[:, cur_lat, cur_lon]
            xCmin = bigMin_ccsm[:, cur_lat, cur_lon]
            xCpr = bigPr_ccsm[:, cur_lat, cur_lon]
            # Append
            xCmax, xCmin, xCpr = np.array(xCmax), np.array(xCmin), np.array(xCpr)
            xCmax_s.append(np.expand_dims(xCmax, axis=1))
            xCmin_s.append(np.expand_dims(xCmin, axis=1))
            xCpr_s.append(np.expand_dims(xCpr, axis=1))
        xCmax_s, xCmin_s, xCpr_s = np.concatenate(xCmax_s, axis=1), np.concatenate(xCmin_s, axis=1), np.concatenate(xCpr_s, axis=1)

        # 2. get G as OUTPUT
        cur_lon, cur_lat = lon[clon], lat[clat]
        g_lat, g_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
        gg_lat, gg_lon = np.where(fGpr.lat == g_lat), np.where(fGpr.lon == g_lon)
        latG, lonG = int(gg_lat[0]), int(gg_lon[0])
        xGpr = bigPr_ground[:, latG, lonG]

        # Convert Data Format to np.array with Shape (sample_num, 1).
        xGpr = np.array(xGpr)
        xGpr = np.expand_dims(xGpr, axis=1)

        # scale.
        scale_ = 1e7
        xCpr_s *= scale_
        xGpr *= scale_

        # Prepare returned data.
        rnt_C = {'Tmax': xCmax_s, 'Tmin': xCmin_s, 'Pr': xCpr_s}
        rnt_G = {'Pr': xGpr}

        return rnt_C, rnt_G

    # Training
    xC, xG = load_data(bigMax_ccsm_tr, bigMin_ccsm_tr, bigPr_ccsm_tr, bigPr_ground_tr)

    # Prediction
    xCp, xGp = load_data(bigMax_ccsm_pr, bigMin_ccsm_pr, bigPr_ccsm_pr, bigPr_ground_pr)

    return (xC, xG, xCp, xGp)


def job_normalization(x):
    for key in x.keys():
        scaler = StandardScaler()
        scaler.fit(x[key])
        x[key] = scaler.transform(x[key])
    return x


def job_concate(data_dict):
    buffer = [data_dict['Tmax'], data_dict['Tmin'], data_dict['Pr']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def concate(data_dict):
    buffer = [data_dict['Pr']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def job_ann_train(datain, dataout):
    ann = MLPRegressor(hidden_layer_sizes=(60, 30), max_iter=1000, tol=1e-4,
                       solver='sgd', activation='relu',
                       learning_rate='adaptive', learning_rate_init=1e-5, alpha=1e-4)
    score = 0
    while score <= 0:
        ann.fit(datain, dataout)
        score = ann.score(datain, dataout)
    return ann, ann.predict(datain), score


def job_ann_predict(ann, datain, dataout):
    score = ann.score(datain, dataout)
    return ann.predict(datain), score


def job_save(name, j, i, lon, lat, xC, xG, xG_ann, xCp, xGp, xGp_ann, myann):
    fname = "./out_prcp/ann_data_prcp." + name
    f = sh.open(fname)
    f['dataid'] = name
    f['arrayindex'] = (j, i)  # index in 0.25mask
    f['latlon'] = (lat, lon)
    f['data_train'] = (xC, xG, xG_ann)
    f['data_predict'] = (xCp, xGp, xGp_ann)
    f['ann'] = myann
    f.close()
    return fname


def workProcess(lat_range, m):
    for j in range(99, 100):
        for i in range(185, 186):
            if m[j, i]:
                # 1. Read data
                Plon, Plat = lon[i], lat[j]
                xC, xG, xCp, xGp = job_readdata(i, j) 
                #print('Read data Finished.')
                continue

                # 2. Normalization
                xC = job_normalization(xC)
                xCp = job_normalization(xCp)
                #print('Normalization Finished.')

                # 3. Concatenate
                xC, xCp, xG, xGp = job_concate(xC), job_concate(xCp), concate(xG), concate(xGp)
                #print('Concatenate Finished.')

                # 4. ANN
                myann, xG_ann, train_score_ann = job_ann_train(xC, xG)
                xGp_ann, test_score_ann = job_ann_predict(myann, xCp, xGp)
                #print('ANN Finished.')

                # 5. Log
                print("Now: j=%3d, i=%3d | ANN: Traning score: %3.3f Testing score: %3.3f" %
                      (j, i, train_score_ann, test_score_ann))

                # 6. Save
                dataid = "%05i" % (j * Nlon + i)
                job_save(dataid, j, i, Plon, Plat, xC, xG, xG_ann, xCp, xGp, xGp_ann, myann)
                #print('Save Data Finished.')
    return


if __name__ == '__main__':
    # 1. Mask
    fmask = xr.open_dataset('../../data/mask_small_0.25_china-tw.nc')
    m = fmask.m
    lon = fmask.lon
    lat = fmask.lat
    Nlat, Nlon = m.shape

    # 2. Load datasets.
    with open('../../data/ccsm4/train_vali/ccsm.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_tr, bigMax_ccsm_pr, bigMin_ccsm_tr, bigMin_ccsm_pr, bigPr_ccsm_tr, bigPr_ccsm_pr = \
      data_dict['bigMax_ccsm_tr'], data_dict['bigMax_ccsm_pr'], \
      data_dict['bigMin_ccsm_tr'], data_dict['bigMin_ccsm_pr'], \
      data_dict['bigPr_ccsm_tr'], data_dict['bigPr_ccsm_pr']

    with open('../../data/ccsm4/train_vali/gmfd_prcp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigPr_ground_tr, bigPr_ground_pr = data_dict['bigPr_ground_tr'], data_dict['bigPr_ground_pr']

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
 
