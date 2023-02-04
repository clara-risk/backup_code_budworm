#coding: utf-8

"""
Summary
-------
Script to produce predictions of SBW outbreak extent from the GAM and RF models. 

References 
----------
PyGAM: https://pygam.readthedocs.io/en/latest/
sklearn: https://scikit-learn.org/stable/index.html
calculating DEV ( deviation from mean elevation ): De Reu et al. (2013), see:
De Reu, J., J. Bourgeois, M. Bats, A. Zwertvaegher, V. Gelorini, P. De Smedt, W. Chu, M. Antrop, P. De Maeyer, P. Finke, M. \
Van Meirvenne, J. Verniers, and P. Crombé. 2013. Application of the topographic position index to heterogeneous landscapes. \
Geomorphology 186.
and:
Gallant, J.C., Wilson, J.P., 2000. Primary topographic attributes. In: Wilson, J.P., Gallant,
J.C. (Eds.), Terrain Analysis: Principles and Applications. Wiley, New York, pp.
51–85.
"""

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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import permutation_importance

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from osgeo import ogr, gdal,osr
from math import floor

from pygam import GAM
from pygam import LogisticGAM, s, f, te, l
from shapely.ops import unary_union

import warnings
warnings.filterwarnings('ignore')

import alphashape

def concave_hull(points,a,compar):
    #You aren't going to run it now, you're going to run it later, after classifying sat image
    alpha = a #for concave hull 

    hull = alphashape.alphashape(points,alpha) #Swith 0 --> alpha
    
    try: 
        if a != 0:  
            hull = MultiPolygon(hull) #[]
        else:
            hull = MultiPolygon([hull])
    except TypeError:
        print('check!') 
        hull = MultiPolygon([hull])
    
    #hull_pts = [poly.exterior.coords.xy for poly in list(hull)]
    if len(hull) != 0: 
        #fig, ax = plt.subplots()
        #ax.scatter(hull_pts[0][0], hull_pts[0][1], color='red',s=1)
        #compar.plot(ax=ax,facecolor='None',edgecolor='red')
        #ax.add_patch(PolygonPatch(hull, fill=False, color='k'))
        #plt.show()

        
        polygon = gpd.GeoDataFrame(index=[0], crs='ESRI:102001', geometry=[hull])  
        #polygon.to_file('data/concave_hull_dbscan/'+name+'.shp', driver='ESRI Shapefile')

        return polygon
    else:
        print('no geometry')
        return []


if __name__ == "__main__":


    year = str(2021)
    yearp = str(2013)


    files = ['100m_on_ndmi_102001_'+year,'asm_'+year,'combo','buff8_'+year,'age','elev','soil_text',\
             '100m_on_ndmi_102001_'+yearp,'100m_on_nbr1_102001_'+year,\
             '100m_on_nbr1_102001_'+yearp,'100m_on_b4b5_102001_'+year,'100m_on_b4b5_102001_'+yearp,\
             'buff100_'+year]
    names = ['ndmi_'+year,'dam','combo','small_buff','age','elev', 'soil_text','ndmi_'+yearp,\
             'nbr1_'+year,'nbr1_'+yearp,'b4b5_'+year,'b4b5_'+yearp,'b100'] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for fi,n in zip(files,names): 
        
        file_name_raster = fi
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
        rb1=src_ds.GetRasterBand(1)
        cols = src_ds.RasterXSize
        cols_list.append(cols)
        rows = src_ds.RasterYSize
        rows_list.append(rows) 
        data = rb1.ReadAsArray(0, 0, cols, rows)
        #data = a[::-1].T[::-1]
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    


    pred['ndmi_diff'] = pred['ndmi_'+yearp] - pred['ndmi_'+year]
    pred['nbr1_diff'] = pred['nbr1_'+yearp] - pred['nbr1_'+year]
    pred['b4b5_diff'] = pred['b4b5_'+yearp] - pred['b4b5_'+year]


    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    print(transformers[0])
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)
    print(lrx)
    print(lry)

    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    
    #mgrid = np.flipud(np.rot90(np.fliplr(np.meshgrid(Xi, Yi)))) #Transpose it? 
    #Xi, Yi = mgrid[:,0].flatten(), mgrid[:,1].flatten()
    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()
    print(len(Xi))
    print(Xi[0]) 
  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any') #.iloc[::100, :]
    
    #df = df[df['buff'] >= 1].dropna(how='any')
    #df = df[df['combo'] > 0] #exclude no species
    #df = df[df['dam'] != 1] #Strange no data value

    df_calc = df 

    print(len(df))
    df = df[df['ndmi_'+year] >= -1]

    df = df[df['ndmi_'+year] <= 1]
    print(len(df))
    df = df[df['ndmi_'+yearp] >= -1]
    print(len(df))
    df = df[df['ndmi_'+yearp] <= 1]
    print(len(df))

    p95 = np.percentile(df_calc['b4b5_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+yearp], 1)
    print(p5)
    df = df[df['b4b5_'+yearp] >= p5]
    df = df[df['b4b5_'+yearp] <= p95]

    p95 = np.percentile(df_calc['b4b5_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+year], 1)
    print(p5)
    df = df[df['b4b5_'+year] >= p5]
    df = df[df['b4b5_'+year] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+yearp], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+yearp], 1)
    print(p5)
    df = df[df['nbr1_'+yearp] >= p5]
    df = df[df['nbr1_'+yearp] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+year], 99)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+year], 1)
    print(p5)
    df = df[df['nbr1_'+year] >= p5]
    df = df[df['nbr1_'+year] <= p95]

    df_mort = df[df['dam'] == 3]
    df_def = df[df['dam'] >= 2]
    df1 = df[df['small_buff'] != 1]
    df_neg = df1[df1['dam'] == 0].sample(n=len(df_def))


    fig, axs = plt.subplots(2, 3, figsize=(7, 7))

    a= sns.histplot(data=df_def, x='ndmi_'+year, kde=True, color="#F39C12", ax=axs[0, 0],edgecolor='None')
    #a.axvline(np.percentile(df_def['ndmi_'+year], 15),c = "#FFB74D")
    #a.axvline(np.percentile(df_def['ndmi_'+year], 70),c = "#FFB74D")
    a = sns.histplot(data=df_neg, x='ndmi_'+year, kde=True, color="#2980B9", ax=axs[0, 0],edgecolor='None')
    #sns.histplot(data=df_mort, x='ndmi_'+year, kde=True, color="#EF5350", ax=axs[0, 0])
    a.set(xlabel='NDMI')
    
    b = sns.histplot(data=df_def, x='nbr1_'+year, kde=True, color="#F39C12", ax=axs[0, 1],edgecolor='None')
    b = sns.histplot(data=df_neg, x='nbr1_'+year, kde=True, color="#2980B9", ax=axs[0, 1],edgecolor='None')
    #b.axvline(np.percentile(df_def['nbr1_'+year], 20),c = "#FFB74D")
    #b.axvline(np.percentile(df_def['nbr1_'+year], 68),c = "#FFB74D")
    #sns.histplot(data=df_mort, x='nbr1_'+year, kde=True, color="#EF5350", ax=axs[0, 1])
    b.set(ylabel=None)
    b.set(xlabel='NBR1')

    c= sns.histplot(data=df_def, x='b4b5_'+year, kde=True, color="#F39C12", ax=axs[0, 2],edgecolor='None')
    c= sns.histplot(data=df_neg, x='b4b5_'+year, kde=True, color="#2980B9", ax=axs[0, 2],edgecolor='None')
    #c.axvline(np.percentile(df_def['b4b5_'+year], 35),c = "#FFB74D")
    #c.axvline(np.percentile(df_def['b4b5_'+year], 80),c = "#FFB74D")
    #sns.histplot(data=df_mort, x='b4b5_'+year, kde=True, color="#EF5350", ax=axs[0, 2])    
    c.set(ylabel=None)
    c.set(xlabel='SWIR / NIR Ratio')
    
    d= sns.histplot(data=df_def, x="ndmi_diff", kde=True, color="#F39C12", ax=axs[1, 0],edgecolor='None')
    d= sns.histplot(data=df_neg, x="ndmi_diff", kde=True, color="#2980B9", ax=axs[1, 0],edgecolor='None')

    #d.axvline(np.percentile(df_def['ndmi_diff'], 40),c = "#FFB74D")
    #d.axvline(np.percentile(df_def['ndmi_diff'], 99.5),c = "#FFB74D")
    #sns.histplot(data=df_mort, x="ndmi_diff", kde=True, color="#EF5350", ax=axs[1, 0])
    #d.set(ylabel=None)
    d.set(xlabel='Difference in NDMI')
    
    e = sns.histplot(data=df_def, x="nbr1_diff", kde=True, color="#F39C12", ax=axs[1,1],edgecolor='None')
    e = sns.histplot(data=df_neg, x="nbr1_diff", kde=True, color="#2980B9", ax=axs[1,1],edgecolor='None')
    #e.axvline(np.percentile(df_def['nbr1_diff'], 45),c = "#FFB74D")
    #e.axvline(np.percentile(df_def['nbr1_diff'], 99.5),c = "#FFB74D")
    #sns.histplot(data=df_mort, x="nbr1_diff", kde=True, color="#EF5350", ax=axs[1,1])
    e.set(ylabel=None)
    e.set(xlabel='Difference in NBR1')
    
    f = sns.histplot(data=df_def, x="b4b5_diff", kde=True, color="#F39C12", ax=axs[1,2],edgecolor='None')
    f = sns.histplot(data=df_neg, x="b4b5_diff", kde=True, color="#2980B9", ax=axs[1,2],edgecolor='None')
    #f.axvline(np.percentile(df_def['b4b5_diff'], 3),c = "#FFB74D")
    #f.axvline(np.percentile(df_def['b4b5_diff'], 50),c = "#FFB74D")
    #sns.histplot(data=df_mort, x="b4b5_diff", kde=True, color="#EF5350", ax=axs[1,2])
    f.set(ylabel=None)
    f.set(xlabel='Difference in SWIR / NIR Ratio')
    
    plt.show()

