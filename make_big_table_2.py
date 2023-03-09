#coding: utf-8

'''
Summary
-------
Script to produce predictions of SBW outbreak extent from the GAM and RF models.

'''

import geopandas as gpd
import pandas as pd 
from geopandas.tools import sjoin
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
from descartes import PolygonPatch
import time
import math
import scipy.stats as stats
import numpy as np
import os, sys
from pyproj import CRS, Transformer
import fiona
import statsmodels.api as sm
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

from osgeo import ogr, gdal,osr

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl

from pygam import GAM
from pygam import LogisticGAM, s, f, te, l
from shapely.ops import unary_union

if __name__ == "__main__":



    files = ['age','sbw_2021','Bf','Sw','Sb','min_temp_jan_daymet','soil_reproj','elev','cent_prox']
    names = ['age','sbw','bf','sw','sb','mj','st','elev','cp'] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for fi,n in zip(files,names): 
        print(fi)
        file_name_raster = fi
        src_ds = gdal.Open('env_covar/final/'+file_name_raster+'.tif')
        rb1=src_ds.GetRasterBand(1)
        cols = src_ds.RasterXSize
        cols_list.append(cols)
        rows = src_ds.RasterYSize
        rows_list.append(rows) 
        data = rb1.ReadAsArray(0, 0, cols, rows)
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    
    pred['age'] = pred['age'] + (2021-2011)
    pred['cp'] = pred['cp']*0.001

    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)


    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()

  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any')

    for nam in names: 

        df = df[df[nam] != -3.4028234663852886e+38]

    df = df.sample(n=2000,random_state=1)

    #print(df)

##
##    fig, ax = plt.subplots(figsize=(15, 15))
##    na_map = gpd.read_file('data/bc_shp.shp')
##    df = df[df['Fire ID'] != 0]
##    sc= plt.scatter(df['lon'],df['lat'],c=df['RBR'],cmap='Spectral_r',s=0.25,alpha=1)
##    na_map.plot(ax=ax, facecolor="none", edgecolor='k',linewidth=1, zorder=14, alpha=1)
##
##    plt.xlabel('Longitude')
##    plt.ylabel('Latitude')
##    plt.xlim(np.min([ulx,lrx]), np.max([ulx,lrx]))
##    plt.ylim(np.min([uly,lry]), np.max([uly,lry]))
##    
##    cb = plt.colorbar(sc)
##    cb.set_label('RBR', rotation=270)
##    plt.show()

    #df = df[df['Fire ID'] != 0]
    df = df[df['st'] != 0]
##    df = df[df['Stand Age 2001'] != 1000]
##    df = df[df['Stand Age 2001'] >= 0]
##    df = df[df['Stand Age 2011'] != 1000]
##    df = df[df['Stand Age 2011'] >= 0]
##    df = df[df['Aspect'] != -9999]
##    df = df[df['Slope'] != -9999]
##    df = df[df['Elevation'] != -9999]
##    df = df[df['Elevation'] != 3000] #No data specified in original raster
##    df = df[df['TPI_500'] != -9999]
##    df = df[df['TPI_1000'] != -9999]
##    df = df[df['TPI_2000'] != -9999]
##    df = df[df['TPI_2500'] != -9999]
##    df = df[df['Prefire Logging'] != -9999]
##    df = df[df['Postfire Logging'] != -9999]
##    df = df[df['Time Since Beetle'] != 1000]
##    df = df[df['MPB Year'] <= 2022]
##    df = df[df['Cause'] != 0]
##    df = df[df['Crown Closure 2001'] >= 0]
##    df = df[df['Crown Closure 2011'] >= 0]
####
##    #df_check = df[df['Crown Closure 2001'] < 0]
##    #print(df_check[['Fire ID','Crown Closure 2001','Crown Closure 2011']])
##
##    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    sns.set_context("paper", font_scale=1.0001)
    z= sns.histplot(data=df, x="age", kde=True, color="r", ax=axs[0, 0],bins=100,edgecolor='None')
    z.set(ylabel=None)
    a = sns.histplot(data=df, x="elev", kde=True, color="r", ax=axs[0, 1],bins=100,edgecolor='None')
    a.set(ylabel=None)

    d = sns.histplot(data=df, x="bf", kde=True, color="r", ax=axs[2,0],bins=100,edgecolor='None')
    d.set(ylabel=None)
    
    sns.histplot(data=df, x="sw", kde=True, color="r", ax=axs[1, 0],bins=100,edgecolor='None')
    #df2 = df
    #df['Fire Size x1000'] = [i/1000 for i in list(df['Fire Size'])]
    b = sns.histplot(data=df, x="sb", kde=False, color="r", ax=axs[1, 1],bins=100,edgecolor='None')
    b.set(ylabel=None) 
    m = sns.histplot(data=df, x="mj", kde=True, color="r", ax=axs[0, 2],bins=100,edgecolor='None')
    m.set(ylabel=None)
    n = sns.histplot(data=df, x="cp", kde=True, color="r", ax=axs[2,1],bins=100,edgecolor='None')
    n.set(ylabel=None)
    c = sns.histplot(data=df, x="st", kde=False, color="r", ax=axs[1, 2],bins=8,edgecolor='None')
    c.set(ylabel=None)
    fig.delaxes(axs[2,2])
    fig.subplots_adjust(hspace=.5,wspace=0.4)
    plt.show()



