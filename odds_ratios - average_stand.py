
#coding: utf-8

"""
Summary
-------
Produce visualizations of the SBW outbreak predictions from the fitted GAM and RF models. 
References 
----------
PyGAM: https://pygam.readthedocs.io/en/latest/
sklearn: https://scikit-learn.org/stable/index.html
"""

import geopandas as gpd
import pandas as pd 
from geopandas.tools import sjoin
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import shape
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
import matplotlib.patches as mpatches
import matplotlib
from osgeo import ogr, gdal,osr
from math import floor

from pygam import GAM
from pygam import LogisticGAM, s, f, te, l

import warnings
warnings.filterwarnings('ignore')

def split_test_train(overall_df):
    '''Split the sample points into training and testing sets 
    Parameters
    ----------
    overall_df : Pandas dataframe 
        Pandas dataframe containing the sample points 
    Returns
    ----------
    Pandas dataframe
        - Two dataframes of the separated training and testing data 
    '''
    train, test = train_test_split(overall_df, test_size=0.3) 
    train = overall_df # For visualization / odds ratio calc purposes 
    return test, train

def auto_odds_discrete(df,range1,range2,split,var,list_control,units,p40,p60,var_name,buffer=10):

    percentiles = []
    vals = []
    vals_under = []
    vals_over = []
    samsize = [] 

    percentiles1 = []
    vals1 = []
    vals_under1 = []
    vals_over1 = []
    samsize1 = [] 

    #ranges = list(range(range1,range2,split)) #+ list(range(55,100,1))
    ranges = [1,2,4,5,6]

    #ranges = list(range(0,53,1)) + list(range(70,100,1))

    for threshold in ranges:
        print(threshold)
        #p = np.percentile(np.array(list(df['combo'])),threshold)
        p = threshold

        #p40 = np.percentile(np.array(list(df[var])),45)
        #print(p40)
        #p40 = 20

        #p60 = np.percentile(np.array(list(df[var])),55)
        #print(p60)
        #p60 = 30


        df['target'] = np.where((df[var] >= (p - buffer)) & (df[var] <= (p + buffer)),1,np.nan)
        df['control'] = np.where((df[var] >= p40) & (df[var] <= p60),1,0)
        df.loc[df.control == 1, 'target'] = 0
        
        df1 = df.dropna(how='any')
        
        X_test = np.array(df1[['target']+list_control])
        y_test = np.array(df1[['sbw']])

        try: 
        
            res = sm.Logit(y_test, X_test).fit()
            res.summary()

            print('odds ratio:')
            print(np.exp(res.params))
            
            conf = res.conf_int()

            if threshold <= p40: 

                vals.append(np.exp(res.params)[0])
                vals_under.append(np.exp(conf)[0][0])
                vals_over.append(np.exp(conf)[0][1])
                percentiles.append(p)
                samsize.append(len(y_test))
            else:
                vals1.append(np.exp(res.params)[0])
                vals_under1.append(np.exp(conf)[0][0])
                vals_over1.append(np.exp(conf)[0][1])
                percentiles1.append(p)
                samsize1.append(len(y_test))
        except np.linalg.LinAlgError:
            if threshold <= p40: 

                vals.append(np.nan)
                vals_under.append(np.nan)
                vals_over.append(np.nan)
                percentiles.append(p)
                samsize.append(len(y_test))
            else:
                vals1.append(np.nan)
                vals_under1.append(np.nan)
                vals_over1.append(np.nan)
                percentiles1.append(p)
                samsize1.append(len(y_test))
            
        
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Calibri"
    #plt.scatter(percentiles,vals,c='r')
    percentiles1 = [str(x) for x in percentiles1]
    percentiles = [str(x) for x in percentiles]
    asymmetric_error = np.array(list(zip(vals_under, vals_over))).T
    plt.errorbar(percentiles,vals,asymmetric_error,fmt='.', c = 'r')
    plt.scatter(percentiles,vals,c='r', label='Odds Ratio')
    #plt.plot(percentiles,vals_under,c='k',linestyle='dashed')
    #plt.plot(percentiles,vals_over,c='k',label='95% Confidence Inverval',linestyle='dashed')

    #plt.scatter(percentiles1,vals1,c='r')
    #plt.plot(percentiles1,vals_under1,c='k',linestyle='dashed')
    #plt.plot(percentiles1,vals_over1,c='k',linestyle='dashed')
    
    asymmetric_error = np.array(list(zip(vals_under1, vals_over1))).T
    plt.errorbar(percentiles1,vals1,asymmetric_error, fmt='.',c = 'r')
    plt.scatter(percentiles1,vals1,c='r')
    
    
    #plt.plot([], [], ' ', label="Comparison Group: " + str(round(p40))+'-' + str(round(p60)) +' '+units)
    plt.axhline(y=1, color='k', linestyle='-')
    plt.xlabel(var_name+' (' +units+')')
    plt.ylabel('Odds Ratio')
    #plt.legend(frameon=False)
    #plt.title('Odds Ratio Plot - Controlling for Covariates')
    a = min(vals_under+vals_under1)-0.2
    b = max(vals_over+vals_over1)+0.2
    ax.set_ylim(a,b)
    plt.ylim([a,b])
    plt.show()

    fin_df = pd.DataFrame()

    fin_df['sample_size'] = samsize+samsize1
    fin_df['percentiles'] = percentiles+percentiles1
    fin_df['odds'] = vals+vals1
    fin_df['upper_odds'] = vals_over+vals_over1
    fin_df['lower_odds'] = vals_under+vals_under1

    fin_df.to_csv(var_name+'.csv',sep=',')


def auto_odds(df,range1,range2,split,var,list_control,units,p40,p60,var_name,buffer=10):

    percentiles = []
    vals = []
    vals_under = []
    vals_over = []
    samsize = [] 

    percentiles1 = []
    vals1 = []
    vals_under1 = []
    vals_over1 = []
    samsize1 = [] 

    ranges = list(range(range1,range2,split)) #+ list(range(55,100,1))

    #ranges = list(range(0,53,1)) + list(range(70,100,1))

    for threshold in ranges:
        print(threshold)
        #p = np.percentile(np.array(list(df['combo'])),threshold)
        p = threshold

        #p40 = np.percentile(np.array(list(df[var])),45)
        #print(p40)
        #p40 = 20

        #p60 = np.percentile(np.array(list(df[var])),55)
        #print(p60)
        #p60 = 30


        df['target'] = np.where((df[var] >= (p - buffer)) & (df[var] <= (p + buffer)),1,np.nan)
        df['control'] = np.where((df[var] >= p40) & (df[var] <= p60),1,0)
        df.loc[df.control == 1, 'target'] = 0
        
        df1 = df.dropna(how='any')
        
        X_test = np.array(df1[['target']+list_control])
        y_test = np.array(df1[['sbw']])

        try: 
        
            res = sm.Logit(y_test, X_test).fit()
            res.summary()

            print('odds ratio:')
            print(np.exp(res.params))
            
            conf = res.conf_int()

            if threshold <= p40: 

                vals.append(np.exp(res.params)[0])
                vals_under.append(np.exp(conf)[0][0])
                vals_over.append(np.exp(conf)[0][1])
                percentiles.append(p)
                samsize.append(len(y_test))
            else:
                vals1.append(np.exp(res.params)[0])
                vals_under1.append(np.exp(conf)[0][0])
                vals_over1.append(np.exp(conf)[0][1])
                percentiles1.append(p)
                samsize1.append(len(y_test))
        except np.linalg.LinAlgError:
            if threshold <= p40: 

                vals.append(np.nan)
                vals_under.append(np.nan)
                vals_over.append(np.nan)
                percentiles.append(p)
                samsize.append(len(y_test))
            else:
                vals1.append(np.nan)
                vals_under1.append(np.nan)
                vals_over1.append(np.nan)
                percentiles1.append(p)
                samsize1.append(len(y_test))
            
        
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Calibri"
    #plt.scatter(percentiles,vals,c='r')
    plt.plot(percentiles,vals,c='r', label='Odds Ratio')
    plt.plot(percentiles,vals_under,c='k',linestyle='dashed')
    plt.plot(percentiles,vals_over,c='k',label='95% Confidence Inverval',linestyle='dashed')

    plt.plot(percentiles1,vals1,c='r')
    plt.plot(percentiles1,vals_under1,c='k',linestyle='dashed')
    plt.plot(percentiles1,vals_over1,c='k',linestyle='dashed')
    
    plt.plot([], [], ' ', label="Comparison Group: " + str(round(p40))+'-' + str(round(p60)) +' '+units)
    plt.axhline(y=1, color='k', linestyle='-')
    plt.xlabel(var_name+' (' +units+'), +/- '+str(buffer)+' Buffer')
    plt.ylabel('Odds Ratio')
    plt.legend(frameon=False)
    #plt.title('Odds Ratio Plot - Controlling for Covariates')
    plt.show()

    fin_df = pd.DataFrame()

    fin_df['sample_size'] = samsize+samsize1
    fin_df['percentiles'] = percentiles+percentiles1
    fin_df['odds'] = vals+vals1
    fin_df['upper_odds'] = vals_over+vals_over1
    fin_df['lower_odds'] = vals_under+vals_under1

    fin_df.to_csv(var_name+'.csv',sep=',')

def byhex(df,referent,vx,vy,vbound_lower,vbound_upper,var,p_low,p_up,units,var_n):

    track_hex = []
    npix = []
    track_rat = []
    track_under = []
    track_over = []

    vx = np.array(vx)
    vy = np.array(vy) 

    hexes = list(set(df['hex']))
    idx = np.isfinite(vx) & np.isfinite(vy)
    e = np.polyfit(vx[idx], vy[idx], 4)
    p = np.poly1d(e)

    x = np.arange(p_low,p_up)
    y = p(x)
    fig, ax = plt.subplots()
    plt.plot(x, y, c='r')
    plt.scatter(vx,vy,c='k')
    plt.xlabel(var_n + ' Value ('+units+')')
    plt.ylabel('Odds Ratio')    
    plt.show()
    fig.savefig(var+'_graph.svg', format='svg', dpi=1300)
    
    for hexcode in hexes:

        #print(hexcode)
        hdf = df[df['hex'] == hexcode] 
        if hexcode not in referent:
            val = float(hdf[var])
            print(val) 
            if val >= vbound_lower and val <= vbound_upper:
                odds = 1 # 1 or 0?
##            elif var == 'mj':
##                odds = p(val)
##                if odds >= 4:
##                    print('check!') 
##                    odds = 4
            else: 
                odds = p(val)
                if odds < 0: 
                    odds = 0
                #if odds > 5:
                    #odds = 5
            print(odds)
            track_rat.append(odds)
             

    fdf = pd.DataFrame()
    fdf['spatial_odds'] = track_rat
##    fdf['npix'] = npix
##    fdf['hexcode'] = track_hex
##    fdf['under'] = track_under
##    fdf['over'] = track_over

    return track_rat
    
def get_comparison(df):
    from shapely.geometry import MultiPoint

    df = df[df['elev'] >= 261]
    df = df[df['elev'] <= 299]
    df = df[df['bf'] <= 2]
    df = df[df['sb'] >= 34]
    df = df[df['sb'] <= 45]
    df = df[df['mj'] <= -25.26]
    df = df[df['mj'] >= -26.37]
    df = df[df['age'] >= 70]
    df = df[df['age'] <= 79]
    #df['coord'] = [Point(xy) for xy in zip(list(df['lon']), df['lat'])]
    wkt_proj = 'PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983",ELLIPSOID["GRS 1980",6378137,298.257222101004,LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Latitude of 1st standard parallel",49,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
    #df['geometry'] = MultiPoint(list(df['coord'].values))
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon, df.lat, crs=wkt_proj),crs=wkt_proj)

    print(df)

    fig,ax = plt.subplots()

    plt.scatter(df['lon'],df['lat'],c='r',s=2)
    plt.show()

    gdf.to_file('average_stand.shp', driver='ESRI Shapefile')

    return list(df['hex']) 
        
if __name__ == "__main__":


    files = ['age','sbw_2021','Bf','Sw','Sb','min_temp_jan_daymet','soil_reproj','elev','cent_prox','hex_test']
    names = ['age','sbw','bf','sw','sb','mj','st','elev','cp','hex'] 
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
        df = df[df[nam] != -9999]

##    df = df.sample(n=2000) 
##    plt.scatter(df['lon'],df['lat'],c=df['hex'])
##    plt.show()

    get_comparison(df)



    
    

    
