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

#only outside buffer
def out_of_area(df,year):

    shp = gpd.read_file('data/'+str(year)+'_ON_fixed.shp')
    p_shp = shp.to_crs('EPSG:32618')
    buff = unary_union([mp.buffer(30*1000) for mp in p_shp['geometry']]) #30 km buffer
    gdf_edge = gpd.GeoDataFrame(crs='EPSG:32618',geometry=[buff])
    gdf_edge = gdf_edge.to_crs('EPSG:4326')
    #gdf_edge.plot(color='red')
    #plt.show()
    
    point = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df['lon'],df['lat']))

    df_new1 = point[~point.geometry.within(gdf_edge['geometry'][0])]

    return df_new1

def concave_hull(points):
    #You aren't going to run it now, you're going to run it later, after classifying sat image
    alpha = 15 #for concave hull 

    hull = alphashape.alphashape(points,alpha) #Swith 0 --> alpha
    print(hull)
    hull = MultiPolygon(hull) #[]
    print(hull)
    hull_pts = [poly.exterior.coords.xy for poly in list(hull)]

    fig, ax = plt.subplots()
    ax.scatter(hull_pts[0][0], hull_pts[0][1], color='red')
    ax.add_patch(PolygonPatch(hull, fill=False, color='green'))
    plt.show()

    crs = {'init': 'epsg:4326'}
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[hull])  
    #polygon.to_file('data/concave_hull_dbscan/'+name+'.shp', driver='ESRI Shapefile')

    return polygon

    
if __name__ == "__main__":

    counter = 0 
    #Make the training data
    training_list = [] 
    for sp_index in ['ndmi']: #,'msr','b4b5']:
        print(sp_index)
        file = 'segmentation_2deriv_'+sp_index+'_2022_11_16_try3.txt'
        df = pd.read_csv(file, delimiter = ",")
        #df.columns = ['','Unnamed: 0','lon','lat','ndmi','niveau','year','dndmi','ymin_ndmi','yleft_ndmi','yright_ndmi'] 
        print(df)
        print(len(df))
        #print(df)
        #print(set(list(df['year'])))
        df = df[df['d'+sp_index] != 'SUP']
        print(len(df))
        df = df[df['d'+sp_index] != 'B']
        print(len(df))
        df = df[df['d'+sp_index] != 'A']
        print(len(df))
        
        df = df[['lon','lat','year','type',sp_index,'ymin_'+sp_index,'yleft_'+sp_index,'yright_'+sp_index,'d'+sp_index]].dropna(how='any')
        counter += len(df)
        train_tracker = []
        print(set(list(df['year'])))
        for year in set(list(df['year'])):
            if year != 'year': 
            
                df_year = df[df['year'] == year]
                

            
                point = gpd.GeoDataFrame(df_year,geometry=gpd.points_from_xy(df_year['lon'],df_year['lat']))
                #print('Computing intersection!') 
                df1 = df_year
                df2 = out_of_area(df_year,year)
                df_train = pd.concat([df1,df2])
                train_tracker.append(df_train)
        df_train = pd.concat(train_tracker)
        training_list.append(df_train)
        

    #ndmi
##
    train = training_list[0]

    train_1 = train[train['type'] == 2]
    train_1['type'] = 1
    print(len(train_1))
    train_0 = train[train['type'] == 0]

    train_0 = train_0.sample(len(train_1))

    train = pd.concat([train_0,train_1]) #balance classses

    print(len(train))
    print((len(train) / counter)*100)

    

    #y_train = np.array(train['type'])
    y_train = train['type']
    print(len(y_train))
##    X_train = np.array(df_train[['ndmi_x','msr','nbr1','b4b5','dndmi_x','dmsr','dnbr1','db4b5','ymin_ndmi_x',\
##                               'ymin_msr','ymin_nbr1','ymin_b4b5','yleft_ndmi_x','yleft_msr','yleft_nbr1','yleft_b4b5',\
##                               'yright_ndmi_x','yright_msr','yright_nbr1','yright_b4b5','year']])
##    X_train = np.array(df_train[['ndmi_x','msr','nbr1','b4b5','dndmi_x','dmsr','dnbr1','ymin_ndmi_x',\
##                               'ymin_msr','ymin_nbr1','yleft_ndmi_x','yleft_msr','yleft_nbr1',\
##                               'yright_ndmi_x','yright_msr','yright_nbr1','year']])
    #X_train = np.array(train[['ndmi','dndmi','ymin_ndmi']])
    #X_train = X_train.astype(np.float)
    X_train = train[['ndmi','dndmi','ymin_ndmi','year']]
    X_train['dndmi'] = [round(float(x),8) for x in X_train['dndmi']]
    X_train['ymin_ndmi'] = [round(float(x)) for x in X_train['ymin_ndmi']]
    #df_vals.to_csv('training_data.csv',sep=',')
    # Fit GAM model

    #Get approx 30% of total points 
    print('Finished formatting data') 
    
##    gam = LogisticGAM(s(0,n_splines=20)+s(1,n_splines=20)+s(2,n_splines=20)+s(3,n_splines=20)+s(4,n_splines=20)\
##                      +s(5,n_splines=20)+s(6,n_splines=20)+s(7,n_splines=20)+s(8,n_splines=20)+s(9,n_splines=20)\
##                      +s(10,n_splines=20)+s(11,n_splines=20)+s(12,n_splines=20)+s(13,n_splines=20)+s(14,n_splines=20)\
##                      +s(15,n_splines=20)+s(16,n_splines=20)+s(17,n_splines=20)+s(18,n_splines=20)+s(19,n_splines=20)\
##                      +s(20,n_splines=20))
##
##    gam = LogisticGAM(s(0,n_splines=20)+s(1,n_splines=20)+s(2,n_splines=20)+s(3,n_splines=20)+s(4,n_splines=20)\
##                  +s(5,n_splines=20)+s(6,n_splines=20)+s(7,n_splines=20)+s(8,n_splines=20)\
##                  +s(9,n_splines=20)+s(10,n_splines=20)+s(11,n_splines=20)+s(12,n_splines=20)\
##                  +s(13,n_splines=20)+s(14,n_splines=20)+s(15,n_splines=20)\
##                  +s(16,n_splines=20))

##    gam = LogisticGAM(s(0,n_splines=30)+s(1,n_splines=30)+s(2,n_splines=30))
##    gam.gridsearch(X_train, y_train)
##    gam.summary()
##
##    fig, axs = plt.subplots(2, 2)
##    titles = ['b5/b4','2nd-deriv','year_min']
##
##    print(axs)
##    count = 0 
##    for i, ax in enumerate(axs.flatten()):
##        if count < len(axs.flatten())-1: 
##            print(i)
##            XX = gam.generate_X_grid(term=i)
##            print(XX)
##            pdep, confi = gam.partial_dependence(term=i, width=.95)
##
##            ax.plot(XX[:, i], pdep)
##            ax.plot(XX[:, i], confi, c='r', ls='--')
##            ax.set_title(titles[i])
##            count+=1 
##
##    plt.show()


    rfc = RandomForestClassifier()
    param_grid = { 
    'max_depth': [30, 50, None],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1,3,5],
    'min_samples_split': [2,20,40,60]
    }   
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    bestF = CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)

    #from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Decision Tree")
    common_params = {
    "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 20,
    "centered": True,
    "random_state": 0,
}
    my_plots = PartialDependenceDisplay.from_estimator(bestF, X_train, ['ndmi','dndmi','ymin_ndmi','year'],ax=ax,kind="both",**common_params)
    
    plt.show()
    #Get the testing data
##
##    for sp_index in ['ndmi']:
##        file = 'segmentation_2deriv_'+sp_index+'_2022_11_16_try3.txt'
##        df = pd.read_csv(file, delimiter = ",")
##        #df.columns = ['','Unnamed: 0','lon','lat','ndmi','niveau','year','dndmi','ymin_ndmi','yleft_ndmi','yright_ndmi'] 
##        print(df[df['year'] == 2020])
##        print(df[df['dndmi'] != 'SUP'])
##        df_test = df[df['dndmi'] != 'SUP']
##
##        df_sup = df[df['dndmi'] == 'SUP']
##        df_test = df_test[['ndmi','dndmi','ymin_ndmi','yleft_ndmi','yright_ndmi','year','lon','lat','type']]
##        df_test = df_test[df_test['ndmi'] != 'ndmi']
##        df_test = df_test[df_test['dndmi'] != 'dndmi']
##        df_test = df_test[df_test['ymin_ndmi'] != 'ymin_ndmi']
##        df_test = df_test[df_test['yright_ndmi'] != 'yright_ndmi']
##        df_test = df_test[df_test['yleft_ndmi'] != 'yleft_ndmi']
##        df_test = df_test[df_test['year'] != 'year']
##
##        df_test = df_test[df_test['type'] != 'type']
##        df_test = df_test[df_test['lon'] != 'lon']
##        df_test = df_test[df_test['lat'] != 'lat']
##        
##        
##        df_test = df_test[['ndmi','dndmi','ymin_ndmi','yleft_ndmi','yright_ndmi','year','lon','lat','type']].dropna(how='any')
##
##    print(len(df_test))
##    y_test = np.array(df_test['type'])
##    y_test = y_test.astype(np.float)
##    X_test = np.array(df_test[['ndmi','dndmi','ymin_ndmi']])     
##    
##    #Zd = gam.predict_proba(X_test)
##
##    classes = list(bestF.classes_)
##    idx = classes.index(1)
##    Zd = bestF.predict_proba(X_test)
##    Zd = [x[idx] for x in Zd]
##    df_test['p_niveau'] = Zd
##
##    df_sup['p_niveau'] = 0  
##
##    df_total = pd.concat([df_sup,df_test])
##    #df_total.to_csv('prediction_11_17_2.csv',sep=',')
##    
##    pred_arr = np.array(df_total['p_niveau'])
##    fig, ax = plt.subplots(figsize=(15, 15))
##    shapefile = 'data/2020_ON_fixed.shp'
##    shp = gpd.read_file(shapefile)
##    shp.plot(ax=ax,facecolor='None',edgecolor='k')
##    #plt.xlabel('Longitude')
##    #plt.ylabel('Latitude')
##    #plt.xticks([])
##    #plt.yticks([])
##    #df_sup = df_sup[df_sup['year'] == 1987]
##    #plt.scatter(df_sup['lon'],df_sup['lat'],c='b',s=0.25)
####    from mpl_toolkits.axes_grid1 import make_axes_locatable
####    divider = make_axes_locatable(ax)
####    cax = divider.append_axes('right', size='5%', pad=0.05)
####    print(df_test)
##    df_test = df_total[df_total['year'] == 2020]
##    print(df_test)
##    #df_test = df_test[df_test['p_niveau'] > 0]
##    plt.scatter(df_test['lon'],df_test['lat'],c=df_test['p_niveau'],s=1,cmap='Spectral')
##    #fig.colorbar(im, cax=cax, orientation='vertical')
##    ax = plt.gca()
##    ax.set_xlim([-96, -75])
##    ax.set_ylim([42, 58])
##    plt.show()
##    
##    pred_arr[pred_arr >= 0.6] = int(1)
##    pred_arr[pred_arr <= 0.6] = int(0)
##    mse = sklearn.metrics.mean_squared_error(np.round(abs(df_total['type'])), np.round(abs(pred_arr)))
##    rmse=math.sqrt(mse)
##    print('RMSE: ' +str(rmse))
##    print(len(pred_arr))
##    
##    matt = sklearn.metrics.matthews_corrcoef(np.round(abs(df_total['type'])), np.round(abs(pred_arr)))
##    print('Matthews correlation coefficient: '+str(matt))
##
##    #points = [Point(x,y) for x, y in zip(df_total['lon'],df_total['lat'])]
##    #gdf = gpd.GeoDataFrame(df_total, crs="EPSG:4326",geometry=gpd.GeoSeries(points))
##    #gdf.to_file("ndmi.geojson", driver="GeoJSON")
##    df_test = df_test[df_test['p_niveau'] >= 0.5]
##    points = [(x,y,) for x,y in zip(df_test.lon, df_test.lat)]
##    ch = concave_hull(points)
##    fig, ax = plt.subplots(figsize=(15, 15))
##    ch.plot(ax=ax,facecolor='None',edgecolor='k')
##    plt.show()
##
##    
