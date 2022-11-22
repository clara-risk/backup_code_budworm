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
    buff = unary_union([mp.buffer(100*1000) for mp in p_shp['geometry']]) #30 km buffer
    gdf_edge = gpd.GeoDataFrame(crs='EPSG:32618',geometry=[buff])
    gdf_edge = gdf_edge.to_crs('EPSG:4326')
    #gdf_edge.plot(color='red')
    #plt.show()
    
    point = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df['lon'],df['lat']))

    df_new1 = point[~point.geometry.within(gdf_edge['geometry'][0])]

    return df_new1

def concave_hull(points,a,compar):
    #You aren't going to run it now, you're going to run it later, after classifying sat image
    alpha = a #for concave hull 

    hull = alphashape.alphashape(points,alpha) #Swith 0 --> alpha
    print(hull)
    if a != 0: 
        hull = MultiPolygon(hull) #[]
    else:
        hull = MultiPolygon([hull])
    print(hull)
    hull_pts = [poly.exterior.coords.xy for poly in list(hull)]
    if len(hull) != 0: 
        #fig, ax = plt.subplots()
        #ax.scatter(hull_pts[0][0], hull_pts[0][1], color='red',s=1)
        #compar.plot(ax=ax,facecolor='blue',edgecolor='red')
        #ax.add_patch(PolygonPatch(hull, fill=False, color='k'))
        #plt.show()

        
        polygon = gpd.GeoDataFrame(index=[0], crs='ESRI:102001', geometry=[hull])  
        #polygon.to_file('data/concave_hull_dbscan/'+name+'.shp', driver='ESRI Shapefile')

        return polygon
    else:
        print('no geometry') 


  
if __name__ == "__main__":

            
    df_total = gpd.read_file('data/concave_hull_test_points_100m.shp')
    df_comp = gpd.read_file('data/2014_subset_proj_concavehull.shp')
    print('read')
    points = [(x,y,) for x, y in zip(df_total['geometry'].x,df_total['geometry'].y)]
    print('format')
    df_total['lon'] = list(df_total['geometry'].x)
    df_total['lat'] = list(df_total['geometry'].y)
    #for pval in list(np.linspace(0,0.05,10)): #0.05 is the upper bound
        #print(pval)
        #ch = concave_hull(points,pval,df_comp)
        #fig, ax = plt.subplots(figsize=(15, 15))
        #df_total = df_total[df_total['p_niveau'] > 0.5]
        #plt.scatter(df_total['lon'],df_total['lat'],c=df_total['p_niveau'],s=1,vmin=0,vmax=1,cmap='Spectral',alpha=0.25)
        #ch.plot(ax=ax,facecolor='None',edgecolor='k')
        #plt.show()

    
    fig, ax = plt.subplots(1,4)
    ax[0].set_title('A.') # 2014 ASM Subset')
    df_comp.plot(ax=ax[0],facecolor='None',edgecolor='red')

    ax[1].set_title('B.') #2014 Regular Point Grid within ASM')
    ax[1].scatter(df_total['lon'],df_total['lat'],c='k',s=0.05)
    df_comp.plot(ax=ax[1],facecolor='None',edgecolor='red')

    ax[2].set_title('C.') #Example of Alpha = 0.014 Concave Hull (Good Fit)')
    df_comp.plot(ax=ax[2],facecolor='None',edgecolor='red')
    ch = concave_hull(points,0.0141,df_comp)
    ch.plot(ax=ax[2],facecolor='None',edgecolor='k')

    ax[3].set_title('D.') # Example of Alpha = 0.001 Concave Hull (Bad Fit)')
    df_comp.plot(ax=ax[3],facecolor='None',edgecolor='red')
    ch2 = concave_hull(points,0.001,df_comp)
    ch2.plot(ax=ax[3],facecolor='None',edgecolor='k')

    ax[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    #ax[0,1].ticklabel_format(useOffset=False, style='plain')
    ax[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    #ax[1,1].ticklabel_format(useOffset=False, style='plain')
    ax[2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    ax[3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    plt.show()
    
    

    
