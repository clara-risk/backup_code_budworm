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
        polygon['AREA'] = (polygon['geometry'].area) *0.000001
        polygon = polygon[polygon['AREA'] >= 3] 
        #polygon.to_file('data/concave_hull_dbscan/'+name+'.shp', driver='ESRI Shapefile')

        return polygon
    else:
        print('no geometry')
        return []

def duplicated(yeari,mod):

    year = str(yeari)
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
        print(fi)
        file_name_raster = fi
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
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
    
    pred['age'] = pred['age'] + (int(year)-2011)

    pred['diff'] = pred['ndmi_'+yearp] - pred['ndmi_'+year]
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
    
    df = df[df['combo'] > 0] #exclude no species

    df_calc = df 

    print(len(df))
    df = df[df['ndmi_'+year] >= -1]

    df = df[df['ndmi_'+year] <= 1]
    print(len(df))
    df = df[df['ndmi_'+yearp] >= -1]
    print(len(df))
    df = df[df['ndmi_'+yearp] <= 1]
    print(len(df))

    p95 = np.percentile(df_calc['b4b5_'+yearp], 95)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+yearp], 5)
    print(p5)
    df = df[df['b4b5_'+yearp] >= p5]
    df = df[df['b4b5_'+yearp] <= p95]

    p95 = np.percentile(df_calc['b4b5_'+year], 95)
    print(p95)
    p5 = np.percentile(df_calc['b4b5_'+year], 5)
    print(p5)
    df = df[df['b4b5_'+year] >= p5]
    df = df[df['b4b5_'+year] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+yearp], 95)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+yearp], 5)
    print(p5)
    df = df[df['nbr1_'+yearp] >= p5]
    df = df[df['nbr1_'+yearp] <= p95]

    p95 = np.percentile(df_calc['nbr1_'+year], 95)
    print(p95)
    p5 = np.percentile(df_calc['nbr1_'+year], 5)
    print(p5)
    df = df[df['nbr1_'+year] >= p5]
    df = df[df['nbr1_'+year] <= p95]


##    fig, axs = plt.subplots(2,3, figsize=(7, 7))
##
##    sns.histplot(data=df, x="ndmi_"+year, kde=True, color="skyblue", ax=axs[0, 0])
##    sns.histplot(data=df, x="nbr1_"+year, kde=True, color="olive", ax=axs[0, 1])
##    sns.histplot(data=df, x="b4b5_"+year, kde=True, color="gold", ax=axs[0,2])
##    sns.histplot(data=df, x="diff", kde=True, color="teal", ax=axs[1, 0])
##    sns.histplot(data=df, x="nbr1_diff", kde=True, color="teal", ax=axs[1, 1])
##    sns.histplot(data=df, x="b4b5_diff", kde=True, color="teal", ax=axs[1, 2])
##    
##    plt.show()
 
    df['dam'] = np.where(df['dam'] >= 2,1,0)
    df_save = df
    
    lengths = []

    trainer = [] 
    for cl in [1,0]: #used to have 3 
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0:
            #num = int(len(df_f) / 1000)
            if len(df_f) >= 2000: 
                num = 2000
                trainer.append(df_f.sample(n=num,random_state=1)) #500000
                lengths.append(num)
            else:
                trainer.append(df_f)
        else:
            #number of negatives varies with cloud mask, etc.
            df_f = df_f[df_f['small_buff'] != 1]
            #num = int(len(df_f) / 1000)
            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1))
            

    print(lengths)
    df2 = pd.concat(trainer)
    df2 = df2.reset_index(drop=True).dropna(how='any')

    fig, axs = plt.subplots(2,3, figsize=(7, 7))

    sns.histplot(data=df2, x="ndmi_"+year, kde=True, color="skyblue", ax=axs[0, 0])
    sns.histplot(data=df2, x="nbr1_"+year, kde=True, color="olive", ax=axs[0, 1])
    sns.histplot(data=df2, x="b4b5_"+year, kde=True, color="gold", ax=axs[0,2])
    sns.histplot(data=df2, x="diff", kde=True, color="teal", ax=axs[1, 0])
    sns.histplot(data=df2, x="nbr1_diff", kde=True, color="teal", ax=axs[1, 1])
    h= sns.histplot(data=df2, x="b4b5_diff", kde=True, color="teal", ax=axs[1, 2])
    h.set(ylabel=None)
    
    plt.show()

    print(set(df2['dam']))
    df_trainX = df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']] #Actually for GAM it has to be an array
    X = np.array(df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]) #,'lat','lon'


    print(len(df_trainX))
    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)
    Y = np.array(df2['dam'])

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    count = 0 

    #bestF = CV_rfc.fit(X, Y)
    #print(CV_rfc.best_params_)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    
    mattc = []
    from sklearn.metrics import matthews_corrcoef
    for train_index, test_index in sss.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print(len(X_train))

        bestF = CV_rfc.fit(X, Y)
        
        Ztest = mod.predict(X_test)
        print(matthews_corrcoef(y_test, Ztest))
        mattc.append(matthews_corrcoef(y_test, Ztest))
        

    df_save['tracker'] = list(range(0,len(df_save))) #index
    rem_track = df_save.dropna(how='any')

    Zi = mod.predict(np.array(rem_track[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]))

    rem_track['pred'] = Zi

    matt = []

    rep = np.array(rem_track['dam'])

    Zn = np.array(rem_track['pred'])
        
    print(confusion_matrix(rep, Zn))

    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(rep, Zn)
    plt.show()

    from sklearn.metrics import matthews_corrcoef
    print(matthews_corrcoef(rep, Zn))
    matt.append(matthews_corrcoef(rep, Zn))

    index_max = max(range(len(matt)), key=matt.__getitem__)

    list_thresh = [0.5]
    thresh = list_thresh[index_max]
    print(thresh)

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])
    total = rem_track

##    fig, ax = plt.subplots(figsize=(15, 15))
##    na_map = gpd.read_file('rasters/temp/2015_proj_clip_dam.shp')
##    na_map = na_map[na_map['DAM'] == 2]
##    rem_track = rem_track[rem_track['pred'] == 1]
##    sc= plt.scatter(rem_track['lon'],rem_track['lat'],c=rem_track['pred'],cmap='Spectral_r',s=0.25,alpha=0.25)
##    na_map.plot(ax=ax, facecolor="none", edgecolor='k',linewidth=1, zorder=14, alpha=1)
##
##    plt.xlabel('Longitude')
##    plt.ylabel('Latitude')
##    plt.xlim(np.min([ulx,lrx]), np.max([ulx,lrx]))
##    plt.ylim(np.min([uly,lry]), np.max([uly,lry]))
##    
##    cb = plt.colorbar(sc)
##    plt.show()
##
##
##    mort = rem_track[rem_track['pred'] == 1]
##    #from scipy.spatial import distance
##    
##    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
##    #points2 = [Point([x,y]) for x, y in zip(mort['lon'],mort['lat'])]
##    #dPoint = gpd.GeoDataFrame(crs='ESRI:102001', geometry=points2)
##    #dPoint.to_file('rasters/concave_hull/2021_test_p.shp', driver='ESRI Shapefile')
##    na_map3 = na_map[na_map['DAM'] ==2]
##    
###    ch = concave_hull(points,0.001405,na_map3)
##    ch2 = concave_hull(points,0.0141,na_map3)
##    
####    fig, ax = plt.subplots(1,2)
######    if len(ch) > 0: 
######        ch.plot(ax=ax[0],facecolor='None',edgecolor='k')
####    if len(ch2) > 0: 
####        ch2.plot(ax=ax[1],facecolor='None',edgecolor='k')
####    na_map3.plot(ax=ax[0],facecolor='red',edgecolor='None',alpha=0.5)
####    na_map3.plot(ax=ax[1],facecolor='red',edgecolor='None',alpha=0.5)
####    plt.show()
##    if len(ch2) > 0: 
##        ch2.to_file('rasters/concave_hull/'+str(yeari)+'_test_y2013.shp', driver='ESRI Shapefile')
        
if __name__ == "__main__":

##    year = list(range(2014,2021+1))
##
##    av_val = [0.72448621,0.68485988,0.695434915,0.710060395,0.712384897,\
##              0.707336253,0.729394809,0.727520442]
##    stdev = [0.01082138,0.009443515,0.017112309,0.010817382,0.015479329,\
##             0.014477619,0.006435234,0.023980813]
##
##    plt.errorbar(year, av_val, stdev, c='#F39C12')
##    plt.scatter(year, av_val, c='#F39C12')
##    plt.xlabel('Year')
##    plt.ylabel('MCC') 
##    plt.show()

    year = list(range(1984,1997+1))
    av_val = [0.732936776,0.765309417,0.705363224,0.744733646,0.731646063,\
              0.712478724,0.749193283,0.766427126,0.783041927,0.770041826,0.766433107,\
              0.708174028,0.702562807,0.694066797]    
    stdev = [0.007975898,0.013159365,0.010641531,0.018480825,0.022009462,0.012613697,\
             0.007191277,0.010371466,0.015960134,0.017634013,0.021983509,0.012695617,\
             0.024622022,0.013109625]

    plt.errorbar(year, av_val, stdev, c='#F39C12')
    plt.scatter(year, av_val, c='#F39C12')
    plt.xlabel('Year')
    plt.ylabel('MCC') 
    plt.show()
