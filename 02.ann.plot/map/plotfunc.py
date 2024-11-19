#!/usr/bin/env python3
#-------------------------------
# Plot 2D Data on Map
# X. Wen and M. Zhang, Apr, 2018

import matplotlib
matplotlib.use('Agg')   # turn off MPL backend to avoid x11 requirement

import numpy                  as np
import xarray                 as xr
import matplotlib.pyplot      as plt
import mpl_toolkits.basemap   as bm
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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
   plt.figure(figsize=paper_a4)

   # Base Map
   #m = bm.Basemap(projection='cyl',resolution='c',
   #               llcrnrlat=0,urcrnrlat=51,
   #               llcrnrlon=70,urcrnrlon=domain[3])
   m = Basemap(llcrnrlat=0,urcrnrlat=51,
               llcrnrlon=80,urcrnrlon=domain[3], 
               projection='lcc',lat_1=33, lat_2=45, lon_0=100)
   m.drawmeridians(np.arange(0,361,10),color='white',
                   linewidth=0.0001,dashes=[999,1],labels=[0,0,0,1])
   m.drawparallels(np.arange(-90,91,10),color='white',
                   linewidth=0.0001,dashes=[999,1],labels=[1,0,0,0])
   m.readshapefile("../../../data/china", 'china', drawbounds=True) 
   m.drawcoastlines()
   m.drawcountries()
   m.readshapefile("../../../data/china_nine_dotted_line", 'nine_dotted', drawbounds=True)
   
   # Plot Data on map
   lon2d,lat2d = np.meshgrid(lon1d,lat1d)
   if len(levs)>1:   fig   = m.contourf(lon2d,lat2d,data2d,levs,latlon=True,cmap=cbar)
   else:             fig   = m.contourf(lon2d,lat2d,data2d,     latlon=True,cmap=cbar)

   # Show color bar
   colorbar = m.colorbar(fig,size='2%')
   #colorbar.set_label(cbarstr)

   # Others
   #plt.title(ti)
   plt.savefig(fout)
   plt.close()
   print('Figure was saved into file: '+fout)

   plt.close()
   return 1
   # END OF plot2d_map


if __name__=='__main__':
   which_year  = 1950   # 1948     to 2010
   which_day   = 15     # 1(Jan/1) to 365(Dec/31)
   levels      = np.arange(-45,46,5)

   f = xr.open_dataset('../data/GMFD/tmax/tmax_0p25_daily_%4i-%4i.nc'%(which_year,which_year),decode_times=False)
   for which_day in range(1,366,15):
      print('\nProcessing ... Year %04i, Day %03i'%(which_year,which_day))
      x = f.tmax.isel(z=0,time=which_day-1)
      x = x-273.15
      plot2d_map( x,x.longitude,x.latitude,levels,
                  ti='GMFD: Julian Day %03i of %04i'%(which_day,which_year),
                  cbarstr='Celsius',
                  fout='figure-%04i-%03i.pdf'%(which_year,which_day) )
