#!/usr/bin/env python3
#-------------------------------
# Plot 2D Data on Map
# X. Wen and M. Zhang, Apr, 2018

import matplotlib
matplotlib.use('Agg')   # turn off MPL backend to avoid x11 requirement

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy                  as np
import xarray                 as xr
import matplotlib.pyplot      as plt
import mpl_toolkits.basemap   as bm

from mpl_toolkits.basemap import Basemap


def plot2d_map(data2d,lon1d,lat1d,levs=[],
               domain=[-90,90,0,360],cbar='jet',cbarstr='',
               ti='',fout='figure.pdf'):
   '''
   ======================================================================
   将2D数据画在map上。输入参数如下：
   data2d      REQUIRED          2D的数据
   lon1d,lat1d REQUIRED          1D的经纬度
   levs        = []              contour levels
   domain      = [-90,90,0,360]  画图区域：南纬度，北纬度，西经度，东经度
   cbar        = 'jet'           画图所用color bar
   cbarstr     = ''              画图在color bar旁边的字符串
   ti          = ''              画图在最上面的title
   fout        = 'figure.pdf'    保存到文件

   by X. Wen and M. Zhang, Peking Univ, Apr 2018
   ======================================================================
   '''

   # Papersize
   paper_a4    = (11.7, 8.27)
   paper_letter= (11,   8.5 )
   fig = plt.figure(figsize=paper_a4)

   # Base Map
   #m = bm.Basemap(projection='cyl',resolution='c',
   #               llcrnrlat=domain[0],urcrnrlat=domain[1],
   #               llcrnrlon=domain[2],urcrnrlon=domain[3])
   m = Basemap(llcrnrlat=0,urcrnrlat=51,
               llcrnrlon=80,urcrnrlon=domain[3], 
               projection='lcc',lat_1=33, lat_2=45, lon_0=100)
   m.drawmeridians(np.arange(0,361,10),color='lightgray',
                   linewidth=0,dashes=[999,1],labels=[0,0,0,1])
   m.drawparallels(np.arange(-90,91,10),color='lightgray',
                   linewidth=0,dashes=[999,1],labels=[1,0,0,0])
   m.readshapefile("../../../data/china", 'china', drawbounds=True) 
   m.readshapefile("../../../data/china_nine_dotted_line", 'nine_dotted', drawbounds=True)
   #m.drawcoastlines()
   #m.drawcountries()
   
   # Plot Data on map
   lon2d,lat2d = np.meshgrid(lon1d,lat1d)
   if len(levs)>1:   fig   = m.contourf(lon2d,lat2d,data2d,levs,latlon=True,cmap=cbar)
   else:             fig   = m.contourf(lon2d,lat2d,data2d,     latlon=True,cmap=cbar)

   # Show color bar
   colorbar = m.colorbar(fig,size='2%')
   colorbar.set_label(cbarstr)

   # Others
   plt.title(ti)
   plt.savefig(fout)
   plt.close()
   print('Figure was saved into file: '+fout)

   plt.close()
   return 1
   # END OF plot2d_map


if __name__=='__main__':
    
    with open('./map_temp.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    yAtr_max, yAtr_min, yCtr_max, yCtr_min, yGtr_max, yGtr_min = \
        data_dict['bigMax_ann'], data_dict['bigMin_ann'], \
        data_dict['bigMax_ccsm'], data_dict['bigMin_ccsm'], \
        data_dict['bigMax_cru'], data_dict['bigMin_cru']

    with open('../../02.bc.plot/map/map_pr.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    yAtr, yCtr, yGtr = \
        data_dict['bigPr_bc'], data_dict['bigPr_ccsm'], data_dict['bigPr_cru']

    #mask
    #fmask = xr.open_dataset('../../../data/mask_small_0.25_china-tw.nc')
    #m = fmask.m
    #lon = fmask.lon
    #lat = fmask.lat
    #Nlat, Nlon = m.shape

    #fCpr =  xr.open_dataset('../../../data/ccsm4/train_vali/ccsm_pr_1948-2005.nc')
    #Lat, Lon = fCpr.lat, fCpr.lon

    #fpr =  xr.open_dataset('../../../../Downscaling-2019-11/data/cru/cru_ts4.01.1901.2016.tmx.dat.nc')
    #Latc, Lonc = fpr.lat, fpr.lon

    #Tmin
    #my.plot2d_map(yGtr_min, Lonc, Latc,
    #              levs = np.arange(-40, 20, 5),
    #              domain = [17, 54, 70, 140],
    #              cbar = 'coolwarm', cbarstr = '',
    #              ti='', fout = './figures/subplot-tmin-cru-2.pdf' )

    my.plot2d_map(yCtr_min - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic',cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ccsm-2.pdf' )

    my.plot2d_map(yAtr_min - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmin-ann-2.pdf' )

    #Tmax
    my.plot2d_map(yGtr_max, Lonc, Latc,
                  levs = np.arange(-5, 45, 5),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-cru-2.pdf' )

    my.plot2d_map(yCtr_max - 273.15, Lon, Lat,
                  levs = np.arange(-14, 15, 1),
                  domain = [0, 51, 80, 120],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ccsm-2.pdf' )

    my.plot2d_map(yAtr_max - 273.15, lon, lat,
                  levs = np.arange(-14, 15, 1), 
                  domain = [17, 54, 70, 140],
                  cbar = 'seismic', cbarstr = '',
                  ti='', fout = './figures/subplot-tmax-ann-2.pdf' )
    #Pr
    my.plot2d_map(yGtr, Lonc, Latc,
                  levs = np.arange(0, 15, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-cru-2.pdf' )

    my.plot2d_map(yCtr, Lon, Lat,
                  levs = np.arange(-14, 14, 1),
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-ccsm-2.pdf' )
 
    my.plot2d_map(yAtr, lon, lat,
                  levs = np.arange(-14, 14, 1), 
                  domain = [17, 54, 70, 140],
                  cbar = 'coolwarm', cbarstr = '',
                  ti='', fout = '../../02.bc.plot/map/figures/subplot-pr-bc-2.pdf' )








