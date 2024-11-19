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
    cc_lat, cc_lon = np.where(fCpr.lat == c_lat), np.where(fCpr.lon == c_lon)
    latc, lonc = int(cc_lat[0]), int(cc_lon[0])

    def load_data(bigMax_ccsm, bigMin_ccsm, bigMax_ground, bigMin_ground):
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

        # 2. get G as OUTPUT
        cur_lon, cur_lat = lon[clon], lat[clat]
        g_lat, g_lon = fGpr.lat.sel(lat=cur_lat, method='nearest'), fGpr.lon.sel(lon=cur_lon, method='nearest')
        gg_lat, gg_lon = np.where(fGpr.lat == g_lat), np.where(fGpr.lon == g_lon)
        latG, lonG = int(gg_lat[0]), int(gg_lon[0])
        xGmax = bigMax_ground[:, latG, lonG]
        xGmin = bigMin_ground[:, latG, lonG]

        # Convert Data Format to np.array with Shape (sample_num, 1).
        xGmax, xGmin = np.array(xGmax), np.array(xGmin)
        xGmax, xGmin = np.expand_dims(xGmax, axis=1), np.expand_dims(xGmin, axis=1)

        # Prepare returned data.
        rnt_C = {'Tmax': xCmax_s, 'Tmin': xCmin_s}
        rnt_G = {'Tmax': xGmax, 'Tmin': xGmin}

        return rnt_C, rnt_G

    # Training
    xC, xG = load_data(bigMax_ccsm_tr, bigMin_ccsm_tr, bigMax_ground_tr, bigMin_ground_tr)

    # Prediction
    xCp, xGp = load_data(bigMax_ccsm_pr, bigMin_ccsm_pr, bigMax_ground_pr, bigMin_ground_pr)

    return (xC, xG, xCp, xGp)


def job_normalization(x):
    for key in x.keys():
        scaler = StandardScaler()
        scaler.fit(x[key])
        x[key] = scaler.transform(x[key])
    return x


def job_concate(data_dict):
    buffer = [data_dict['Tmax'], data_dict['Tmin']]
    data_dict = np.concatenate(buffer, axis=1)
    return data_dict


def add_julian(data_dict, sin):
    nd_, ndim_ = data_dict.shape
    ny_ = nd_ //365
    new_sin = sin
    count = 0
    while count <= ny_-2:
        new_sin = np.row_stack((new_sin, sin))
        count = count + 1
    new_sin = new_sin.reshape(-1)    
    new_sin = np.array(new_sin)
    data = np.c_[data_dict, new_sin]
    return data


def job_ann_train(datain, dataout):
    ann = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=100, tol=1e-4, 
                       solver='sgd', activation='relu', 
                       learning_rate='adaptive', learning_rate_init=1e-5, alpha=1e-4)
    score = 0
    while score <= 0.0001:
        ann.fit(datain, dataout)
        score = ann.score(datain, dataout)
    return ann, ann.predict(datain), score


def job_ann_predict(ann, datain, dataout):
    score = ann.score(datain, dataout)
    return ann.predict(datain), score


def job_save(name, j, i, lon, lat, xC, xG, xG_ann, xCp, xGp, xGp_ann, myann):
    fname = "./out_temp/ann_data_temp." + name
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

                # 2. Normalization
                xC = job_normalization(xC)
                xCp = job_normalization(xCp)
                #print('Normalization Finished.')

                # 3. Concatenate
                xC, xCp, xG, xGp = job_concate(xC), job_concate(xCp), job_concate(xG), job_concate(xGp)
                #print('Concatenate Finished.')

                # 4. Add Julian
                #xC, xCp = add_julian(xC, e), add_julian(xCp, e)
                #print('Add Julian Finished. ')

                # 5. ANN
                myann, xG_ann, train_score_ann = job_ann_train(xC, xG)
                xGp_ann, test_score_ann = job_ann_predict(myann, xCp, xGp)
                #print('ANN Finished.')

                # 6. Log
                print("Now: j=%3d, i=%3d | ANN: Traning score: %3.3f Testing score: %3.3f" %
                      (j, i, train_score_ann, test_score_ann))
                continue

                # 7. Save
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

    # 3. Load datasets.
    with open('../../data/ccsm4/train_vali/ccsm.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ccsm_tr, bigMax_ccsm_pr, bigMin_ccsm_tr, bigMin_ccsm_pr = \
      data_dict['bigMax_ccsm_tr'], data_dict['bigMax_ccsm_pr'], \
      data_dict['bigMin_ccsm_tr'], data_dict['bigMin_ccsm_pr']
    #  data_dict['bigPr_ccsm_tr'], data_dict['bigPr_ccsm_pr']

    with open('../../data/ccsm4/train_vali/gmfd_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ground_tr, bigMax_ground_pr = data_dict['bigMax_ground_tr'], data_dict['bigMax_ground_pr']

    with open('../../data/ccsm4/train_vali/gmfd_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ground_tr, bigMin_ground_pr = data_dict['bigMin_ground_tr'], data_dict['bigMin_ground_pr']

    # 4. Multi_process
    process_num =1
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
 
