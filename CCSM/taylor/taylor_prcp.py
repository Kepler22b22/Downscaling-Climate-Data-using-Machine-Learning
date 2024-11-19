# Zhang MQ, Nov 2019


import os
import matplotlib.pyplot as plt
import numpy             as np
import pickle
import skill_metrics     as sm

from matplotlib            import rcParams
from sys                   import version_info
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class Container(object): 
    
    def __init__(self, pred1, pred2, pred3, ref):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.ref = ref


def get_average(bigArray):
    m_aa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m_bb = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    nd_ = bigArray.size
    ny_ = nd_ // 365
    meanArray = np.zeros((ny_, 12), dtype='float32')
    bigArray = bigArray.reshape(ny_, 365)
    for i in range(12):
        meanArray[:, i] = np.mean(bigArray[:, m_aa[i]:m_bb[i]], axis=1)
    return meanArray


if __name__ == '__main__':
    # Set the figure properties (optional)
    rcParams["figure.figsize"] = [10.0, 8.0]
    rcParams['lines.linewidth'] = 1    # line width for plots
    rcParams.update({'font.size': 12}) # font size of axes text

    city = ['Harbin', 'Shanghai', 'Guangdong', 'Chengdu', 'Lasa', 'Urumuchi', 'Lanzhou']

    for i in range(0, 7):
        data = load_obj('./city/prcp/' + city[i] + '_talyor_ccsm')
        print(city[i])

        # Calculate statistics for Taylor diagram
        # The first array element (e.g. taylor_stats1[0]) corresponds to the
        # reference series while the second and subsequent elements
        # (e.g. taylor_stats1[1:]) are those for the predicted series.
        taylor_stats1 = sm.taylor_statistics(data['bigPr_ann_his'], data['bigPr_cru_his'], 'data')
        taylor_stats2 = sm.taylor_statistics(data['bigPr_lr_his'], data['bigPr_cru_his'], 'data')
        taylor_stats3 = sm.taylor_statistics(data['bigPr_bc_his'], data['bigPr_cru_his'], 'data') 
        taylor_stats4 = sm.taylor_statistics(data['bigPr_ccsm_his'], data['bigPr_cru_his'], 'data')
        taylor_stats5 = sm.taylor_statistics(data['bigPr_poisson_his'], data['bigPr_cru_his'], 'data')

        # Store statistics in arrays
        sdev_prcp = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1],
                              taylor_stats2['sdev'][1], taylor_stats3['sdev'][1], 
                              taylor_stats4['sdev'][1], taylor_stats5['sdev'][1]]) 
        crmsd_prcp = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1],
                               taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1], 
                               taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1]]) 
        ccoef_prcp = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1],
                               taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], 
                               taylor_stats4['ccoef'][1], taylor_stats5['ccoef'][1]])

        print(sdev_prcp)
        print(crmsd_prcp)
        print(ccoef_prcp)

        sm.taylor_diagram(sdev_prcp, crmsd_prcp, ccoef_prcp, alpha=0.0, tickRMSangle = 115,
                          tickRMS = [0, 1, 2, 3, 4], 
                          markerLabel = ['None', 'ANN', 'Linear', 'SD', 'CCSM', 'Poisson'],
                          markerLegend = 'on', markerSize = 12,  
                          styleOBS = '-', colOBS = 'r', widthOBS = 1.5, titleOBS = 'Ref', 
                          widthSTD = 1.5, widthCOR = 1.5, 
                          checkstats = 'on')

        # Write plot to file
        plt.savefig('./figures/taylor_prcp_' + city[i] + '.png')
        plt.close()









