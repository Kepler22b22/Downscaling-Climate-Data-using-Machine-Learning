# Zhang MQ, Jun 2020

import os
import pickle

import numpy  as np
import pandas as pd
import xarray as xr
import shelve as sh

import matplotlib
matplotlib.use('Agg')  # turn off MPL backend to avoid x11 requirement
import matplotlib.pyplot    as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def get_alpha(big_ann, big_ground):
	alpha_new = np.zeros((12))
	for i in range(12):
		big_ann_tmp = big_ann[:, m_aa[i]:m_bb[i]].reshape(-1)
		big_ground_tmp = big_ground[:, m_aa[i]:m_bb[i]].reshape(-1)
		alpha_new[i] = np.std(big_ground_tmp) / np.std(big_ann_tmp)
	return alpha_new


#def get_alpha(big_ann_tr, big_ground_tr):
#    alpha_new = np.zeros((12))
#    for i in range(12):
#        for j in range(14):
#            big_ann = big_ann_tr[j, m_aa[i]:m_bb[i]].reshape(-1)
#            big_ground = big_ground_tr[j, m_aa[i]:m_bb[i]].reshape(-1)
#            alpha_new[j, i] = np.std(big_ground) / np.std(big_ann)
#    return alpha_new


def postwork(alpha, big_ann_tr, big_ann_pr, big_ground_tr):
    ny_, nd_ = big_ann_pr.shape
    big_ann = np.zeros((ny_, 365))
    for i in range(12):
        mean_ann_tr = np.mean(big_ann_tr[:, m_aa[i]:m_bb[i]])
        mean_ann_pr = np.mean(big_ann_pr[:, m_aa[i]:m_bb[i]])
        mean_ground_tr = np.mean(big_ground_tr[:, m_aa[i]:m_bb[i]])
        big_ann[:, m_aa[i]:m_bb[i]] = ((big_ann_pr[:, m_aa[i]:m_bb[i]] - mean_ann_pr) * alpha[i]) + \
                                          (mean_ann_pr - mean_ann_tr + mean_ground_tr)
    return big_ann


if __name__ == '__main__':

    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365] 

    # Load Data
    with open('./China_historical_tmax_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ann_his = \
      data_dict['bigMax_ann_historical']
 
    with open('./China_historical_tmin_raw.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ann_his = \
      data_dict['bigMin_ann_historical']

    with open('../../../data/ccsm4/future/tr_tmax.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMax_ground_his = data_dict['bigMax_ground_tr']

    with open('../../../data/ccsm4/future/tr_tmin.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    bigMin_ground_his = data_dict['bigMin_ground_tr']

    bigMax_ann_his, bigMin_ann_his = \
      bigMax_ann_his[:, :, 99, 185], bigMin_ann_his[:, :, 99, 185]

    bigMax_ground_his, bigMin_ground_his = \
      bigMax_ground_his[:, 99, 185].values.reshape(56, 365), bigMin_ground_his[:, 99, 185].values.reshape(56, 365)
    
    alpha_tmax = get_alpha(bigMax_ann_his, bigMax_ground_his)
    alpha_tmin = get_alpha(bigMin_ann_his, bigMin_ground_his)

    bigMax_ann_his, bigMin_ann_his = \
      postwork(alpha_tmax, bigMax_ann_his, bigMax_ann_his, bigMax_ground_his), \
      postwork(alpha_tmin, bigMin_ann_his, bigMin_ann_his, bigMin_ground_his)

    with open('Beijing_his_temp_postwork.pkl', 'wb') as f:
        pickle.dump({'bigMax_ann_his': bigMax_ann_his,
                     'bigMin_ann_his': bigMin_ann_his,
                     'alpha_tmax': alpha_tmax,
                     'alpha_tmin': alpha_tmin}, f)




