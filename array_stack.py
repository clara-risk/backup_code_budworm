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


    files = ['ndmi_2014_corr','asm_raster_2014','combo_raster_2014','buff_30km_2014_corr','buff_8km_2014','bf_2014','age_2014']
    names = ['ndmi','dam','combo','buff','small_buff','bf','age'] 
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
        #data = a[::-1].T[::-1]
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    
    
    pred['year'] = np.ones(np.shape(pred['ndmi']))+2014
    pred['age'] = pred['age'] + (2014-2011)
    
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

    
    df = pd.DataFrame(pred)
    
    df = df[df['buff'] >= 1].dropna(how='any')
    df = df[df['combo'] != 0] #exclude no species
    df = df[df['dam'] != 1] #Strange no data value
    df_save = df 
    #Distance Matrix

    #Get the pixel centroids of initiation
##    cent = gpd.read_file('outputs/final/proj/temp/centroids_of_initiation.shp')
##    aff_x = cent['geometry'].x
##    aff_y = cent['geometry'].y
##
##    from scipy.spatial import distance
##    from scipy.spatial import cKDTree
##    
##    points = [(x,y,) for x, y in zip(aff_x,aff_y)]
##    ap = [(x,y,) for x, y in zip(df['lon'],df['lat'])]
##    #mdist = distance.cdist(np.array(points), np.array(points), 'euclidean')
##    #distm = np.amin(mdist, axis=1)
##    min_dists, min_dist_idx = cKDTree(np.array(points)).query(np.array(ap), 1)
##    
##    df['spatial_auto'] = min_dists
    #df['spatial_auto'] = np.where(df['small_buff'].eq(1), 0, df['spatial_auto']) #set equal probability inside 8km buff --> no overfits
    #Done Distance Matrix

    #Just checking!!

##    fig, ax = plt.subplots(figsize=(15, 15))
##    na_map = gpd.read_file('data/2014_ON_fixed_proj102001.shp')
##    
##    #crs = {'init': 'epsg:4326'}
##    
##    dfs = df[df['dam'] >= 0]
##    sc= plt.scatter(dfs['lon'],dfs['lat'],c=dfs['spatial_auto'],cmap='plasma',s=1,alpha=0.25) #c=dam
##    na_map.plot(ax=ax, facecolor="none", edgecolor='k',linewidth=1, zorder=14, alpha=1)
##    plt.xlabel('Longitude')
##    plt.ylabel('Latitude')
##    #plt.xlim(np.min([ulx,lrx]), np.max([ulx,lrx]))
##    #plt.ylim(np.min([uly,lry]), np.max([uly,lry]))
##    
##    cb = plt.colorbar(sc)
##    plt.show()
    
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
    df2 = pd.concat(trainer)
    
    print(set(df2['dam']))
    df_trainX = df2[['ndmi','bf','age']]
    print(len(df_trainX))
    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)


    rfc = BalancedRandomForestClassifier()
    param_grid = { 
    'max_depth': [5],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1],
    'min_samples_split': [2]
    }   
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

    #reg = BalancedRandomForestClassifier(max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 20, random_state=1)
    #reg.fit(df_trainX, df_trainY)
    bestF = CV_rfc.fit(df_trainX, df_trainY)
    print(CV_rfc.best_params_)
    
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
##    my_plots = PartialDependenceDisplay.from_estimator(reg, df_trainX, ['ndmi','bf','age','spatial_auto'],target=2,ax=ax,kind="both",**common_params)
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
##    my_plots = PartialDependenceDisplay.from_estimator(reg, df_trainX, ['ndmi','bf','age','spatial_auto'],target=3,ax=ax,kind="both",**common_params)
##    plt.show()
    

    df_save['tracker'] = list(range(0,len(df_save))) #index
    rem_track = df_save.dropna(how='any')
    #rem_track = rem_track[rem_track['small_buff'] == 1]
    print(len(rem_track))
    #rem_track = rem_track.iloc[::10, :]
    Zi = bestF.predict(rem_track[['ndmi','bf','age']])

    rem_track['pred'] = Zi

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])

    fig, ax = plt.subplots(figsize=(15, 15))
    na_map = gpd.read_file('data/2014_ON_fixed_proj102001.shp')

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
    from scipy.spatial import distance
    
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    
##    mdist = distance.cdist(np.array(points), np.array(points), 'euclidean')
##    distm = np.amax(mdist, axis=1)
##    tdf = pd.DataFrame()
##    tdf['d'] = list(distm)
##    tdf['points'] = points
##    tdf = tdf[tdf['d'] >= 4000]
##    points = tdf['points']

    
    na_map3 = na_map[na_map['DAM'] == 3]
    
    ch = concave_hull(points,0.001,na_map3)
    ch2 = concave_hull(points,0.0141,na_map3)

    fig, ax = plt.subplots(1,2)
    if len(ch) > 0: 
        ch.plot(ax=ax[0],facecolor='None',edgecolor='k')
    if len(ch2) > 0: 
        ch2.plot(ax=ax[1],facecolor='None',edgecolor='k')
    na_map3.plot(ax=ax[0],facecolor='red',edgecolor='None',alpha=0.5)
    na_map3.plot(ax=ax[1],facecolor='red',edgecolor='None',alpha=0.5)
    plt.show()
    

    mort = rem_track[rem_track['pred'] == 2]
    
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    na_map3 = na_map[na_map['DAM'] == 2]
    
    ch = concave_hull(points,0.001,na_map3)
    ch2 = concave_hull(points,0.0141,na_map3)

    fig, ax = plt.subplots(1,2)
    if len(ch) > 0: 
        ch.plot(ax=ax[0],facecolor='None',edgecolor='k')
    if len(ch2) > 0: 
        ch2.plot(ax=ax[1],facecolor='None',edgecolor='k')
    na_map3.plot(ax=ax[0],facecolor='red',edgecolor='None',alpha=0.5)
    na_map3.plot(ax=ax[1],facecolor='red',edgecolor='None',alpha=0.5)
    plt.show()
    
    
    
    
