
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
        except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
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


    #auto_odds(df,20,160,5,'age',['cp','bf','sw','sb','mj','st','elev'],'years',0,20,'Stand Age')

    #auto_odds(df,20,int(max(df['bf'])-10),5,'bf',['cp','age','sw','sb','mj','st','elev'],'%',0,20,'Balsam Fir Abundance')

    #auto_odds(df,200,int(max(df['elev'])-10),5,'elev',['cp','age','sw','sb','mj','st','bf'],'m',0,200,'Elevation')

    #auto_odds(df,0,400,30,'elev',['cp','age','sw','sb','mj','st','bf'],'m',400,int(max(df['elev'])-30),'Elevation',buffer=30)

    #auto_odds(df,200,int(max(df['elev'])-20),10,'elev',['cp','age','sw','sb','mj','st','bf','lat','lon'],'m',0,200,'Elevation',buffer=20)

    #df = df.sample(n = 15000,random_state=1)
    
##    p40 = np.percentile(np.array(list(df['bf'])),45)
##    p60 = np.percentile(np.array(list(df['bf'])),55)
##    print(p60)
##    auto_odds(df,0,int(max(df['bf'])-10),5,'bf',['cp','age','sw','sb','mj','st','elev','lat','lon'],'%',p40,p60,'Balsam Fir',buffer=5)

##    p40 = np.percentile(np.array(list(df['elev'])),45)
##    p60 = np.percentile(np.array(list(df['elev'])),55)
##    auto_odds(df,0,550,20,'elev',['cp','age','sw','sb','mj','st','bf','lat','lon'],'m',p40,p60,'Elevation',buffer=20)

    #df = df.sample(n=2000,random_state=1)
##    p40 = np.percentile(np.array(list(df['mj'])),45)
##    p60 = np.percentile(np.array(list(df['mj'])),55)
##    print(p40)
##    print(p60)
##    auto_odds(df,-28,-16,1,'mj',['cp','age','sw','sb','bf','st','elev','lat','lon'],'°C',p40,p60,\
##              'January Minimum Temperature',buffer=4)

##    
##    p40 = np.percentile(np.array(list(df['mj'])),45)
##    p60 = np.percentile(np.array(list(df['mj'])),55)
##    print(p40)
##    print(p60)
##    auto_odds(df,-28,-16,1,'mj',['cp','age','sw','sb','bf','st','elev','lat','lon'],'°C',p40,p60,\
##              'January Minimum Temperature',buffer=4)

    df['mj_abs'] = df['mj']

    p40 = np.percentile(np.array(list(df['mj_abs'])),40)
    p60 = np.percentile(np.array(list(df['mj_abs'])),60)
    print(p40)
    print(p60)
    auto_odds(df,-29,-16,1,'mj_abs',['cp','age','sw','sb','bf','st','lon'],'°C',p40,p60,\
              'January Minimum Temperature',buffer=2)#'st','elev','lat','lon'

##    df['cp'] = df['cp']*0.001
##
##    p40 = np.percentile(np.array(list(df['cp'])),0)
##    p60 = np.percentile(np.array(list(df['cp'])),1)
##    print(p40)
##    print(p60)
##    auto_odds(df,0,300,5,'cp',['mj','age','sw','sb','bf','st','elev'],'km',0,5,\
##              'Distance from Outbreak Origin',buffer=5)


