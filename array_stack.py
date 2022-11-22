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
    print(hull)
    try: 
        if a != 0:  
            hull = MultiPolygon(hull) #[]
        else:
            hull = MultiPolygon([hull])
    except TypeError:
        hull = MultiPolygon([hull])
    
    hull_pts = [poly.exterior.coords.xy for poly in list(hull)]
    if len(hull) != 0: 
        fig, ax = plt.subplots()
        #ax.scatter(hull_pts[0][0], hull_pts[0][1], color='red',s=1)
        compar.plot(ax=ax,facecolor='blue',edgecolor='red')
        ax.add_patch(PolygonPatch(hull, fill=False, color='k'))
        plt.show()

        
        polygon = gpd.GeoDataFrame(index=[0], crs='ESRI:102001', geometry=[hull])  
        #polygon.to_file('data/concave_hull_dbscan/'+name+'.shp', driver='ESRI Shapefile')

        return polygon
    else:
        print('no geometry') 


if __name__ == "__main__":


    files = ['ndmi_raster','asm_raster','combo_raster','buff_30km','buff_8km']
    names = ['ndmi','dam','combo','buff','small_buff'] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for f,n in zip(files,names):
        
        file_name_raster = f
        src_ds = gdal.Open('outputs/final/proj/'+file_name_raster+'.tif')
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
    
    
    pred['year'] = np.ones(np.shape(pred['ndmi']))+2014
    
    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]

    lrx = ulx + (row_num * abs(xres))
    lry = uly + (col_num * abs(yres))


    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    
    mgrid = np.rot90(np.fliplr(np.meshgrid(Xi, Yi))) #Transpose it? 
    Xi, Yi = mgrid[:,0].flatten(), mgrid[:,1].flatten()
    print(Xi[0])
    print(Yi[0]) 
  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred)
    
    df = df[df['buff'] >= 1].dropna(how='any')
    df_save = df 
    df = df[df['combo'] != 0] #exclude no species
    #df = df[df['dam'] != 0].dropna(how='any')

    #Just checking!!

    fig, ax = plt.subplots(figsize=(15, 15))
    na_map = gpd.read_file('data/2014_ON_fixed_proj102001.shp')
    
    crs = {'init': 'epsg:4326'}
    
    dfs = df[df['dam'] == 2]
    sc= plt.scatter(dfs['lon'],dfs['lat'],c=dfs['dam'],cmap='plasma',s=1,alpha=0.25)
    na_map.plot(ax=ax, facecolor="none", edgecolor='k',linewidth=1, zorder=14, alpha=1)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(np.min([ulx,lrx]), np.max([ulx,lrx]))
    plt.ylim(np.min([uly,lry]), np.max([uly,lry]))
    
    cb = plt.colorbar(sc)
    plt.show()
    
    lengths = []

    trainer = [] 
    for cl in [0,2,3]:
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0: 
            trainer.append(df_f)
        else:
            df_f = df_f[df_f['small_buff'] != 1]
            trainer.append(df_f.sample(frac=0.1,random_state=1))
            
        lenf = len(df_f)
        lengths.append(lenf)

    mlen = min(lengths)
    print(lengths)
    df = pd.concat(trainer)
    
    print(set(df['dam']))
    df_trainX = df[['ndmi','year']]
    print(len(df_trainX))
    df_trainY = np.array(df[['dam']]).reshape(-1, 1)

    reg = BalancedRandomForestClassifier(max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 20, random_state=1)
    reg.fit(df_trainX, df_trainY)
    
##    from sklearn.inspection import PartialDependenceDisplay
##    
##    fig, ax = plt.subplots(figsize=(12, 6))
##    ax.set_title("Decision Tree, Moderate-to-Severe")
##    common_params = {
##    "subsample": 50,
##    "n_jobs": 2,
##    "grid_resolution": 20,
##    "centered": True,
##    "random_state": 1}
##    my_plots = PartialDependenceDisplay.from_estimator(reg, df_trainX, ['ndmi','year','lat','lon'],target=2,ax=ax,kind="both",**common_params)
##    plt.show()
##
##    fig, ax = plt.subplots(figsize=(12, 6))
##    ax.set_title("Decision Tree, Mortality")
##    common_params = {
##    "subsample": 50,
##    "n_jobs": 2,
##    "grid_resolution": 20,
##    "centered": True,
##    "random_state": 1}
##    my_plots = PartialDependenceDisplay.from_estimator(reg, df_trainX, ['ndmi','year','lat','lon'],target=3,ax=ax,kind="both",**common_params)
##    plt.show()
    

    df_save['tracker'] = list(range(0,len(df_save))) #index
    rem_track = df_save.dropna(how='any')
    rem_track = rem_track[rem_track['small_buff'] == 1]
    print(len(rem_track))
    #rem_track = rem_track.iloc[::500, :]
    Zi = reg.predict(rem_track[['ndmi','year']])

    rem_track['pred'] = Zi

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])

    fig, ax = plt.subplots(figsize=(15, 15))
    na_map = gpd.read_file('data/2014_ON_fixed_proj102001.shp')
    
    crs = {'init': 'epsg:4326'}

    #rem_track = rem_track[rem_track['pred'] > 0]
    

    sc= plt.scatter(rem_track['lon'],rem_track['lat'],c=rem_track['pred'],cmap='Spectral_r',s=1,alpha=0.25)
    na_map.plot(ax=ax, facecolor="none", edgecolor='k',linewidth=1, zorder=14, alpha=1)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(np.min([ulx,lrx]), np.max([ulx,lrx]))
    plt.ylim(np.min([uly,lry]), np.max([uly,lry]))
    
    cb = plt.colorbar(sc, spacing='proportional',ticks=[0,2,3])
    plt.show()


    mort = rem_track[rem_track['pred'] == 3]
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    na_map3 = na_map[na_map['DAM'] == 3]
    
    ch = concave_hull(points,0.0141,na_map3)
    

    
    
    
