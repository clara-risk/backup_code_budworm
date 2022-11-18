     

#coding: utf-8

"""
Summary
-------
Analysis to determine R2 cutoff for supervised classification component of algorithm. 

References 
----------

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
#import fiona
import statsmodels.api as sm

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib
#from osgeo import ogr, gdal,osr
from math import floor
from shapely.ops import unary_union

import warnings
warnings.filterwarnings('ignore')
import pathlib
import seaborn as sns
import itertools

if __name__ == "__main__":



    df_b4b5_aff = pd.read_csv('b4b5_affected.txt',sep=',')
    r2_b4b5_aff = list(df_b4b5_aff['b4b5'])

    df_b4b5_unaff = pd.read_csv('b4b5_unaffected.txt',sep=',')
    r2_b4b5_unaff = list(df_b4b5_unaff['b4b5'])


    df_nbr1_unaff = pd.read_csv('nbr1_unaffected.txt',sep=',')
    r2_nbr1_unaff = list(df_nbr1_unaff['nbr1'])

    df_nbr1_aff = pd.read_csv('nbr1_affected.txt',sep=',')
    r2_nbr1_aff = list(df_nbr1_aff['nbr1'])
    
    df_ndmi_aff = pd.read_csv('ndmi_affected.txt',sep=',')
    r2_ndmi_aff = list(df_ndmi_aff['ndmi'])

    df_ndmi_unaff = pd.read_csv('ndmi_unaffected.txt',sep=',')
    r2_ndmi_unaff = list(df_ndmi_unaff['ndmi'])

    df_msr_aff = pd.read_csv('msr_affected.txt',sep=',')
    r2_msr_aff = list(df_msr_aff['msr'])
    
    df_msr_unaff = pd.read_csv('msr_unaffected.txt',sep=',')
    r2_msr_unaff = list(df_msr_unaff['msr'])

    df = pd.DataFrame()
    #df['idx'] = list(range(0,len(r2_b4b5_aff)+len(r2_b4b5_unaff)\
                           #+len(df_nbr1_aff)+len(df_nbr1_unaff)))
    
    df['r2'] = r2_b4b5_aff+r2_b4b5_unaff + r2_nbr1_aff + r2_nbr1_unaff + \
               r2_ndmi_aff + r2_ndmi_unaff + r2_msr_aff + r2_msr_unaff
    
    df['Type'] = ['Affected']*len(r2_b4b5_aff) + ['Unaffected']*len(r2_b4b5_unaff)+\
                 ['Affected']*len(r2_nbr1_aff) + ['Unaffected']*len(r2_nbr1_unaff)+\
                 ['Affected']*len(r2_ndmi_aff) + ['Unaffected']*len(r2_ndmi_unaff)+\
                 ['Affected']*len(r2_msr_aff) + ['Unaffected']*len(r2_msr_unaff)
    df['Spectral Index'] = ['B4B5 Ratio']*(len(r2_b4b5_aff)+len(r2_b4b5_unaff))+\
                ['NBR1']*(len(r2_nbr1_aff)+len(r2_nbr1_unaff))+\
                ['NDMI']*(len(r2_ndmi_aff)+len(r2_ndmi_unaff))+\
                ['MSR']*(len(r2_msr_aff)+len(r2_msr_unaff))
    ax = sns.boxplot(data=df, x="Spectral Index", y="r2",hue='Type', palette="Spectral")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("$R^2$",size=12)
    ax.set_xlabel("Trend in Spectral Index",size=12)
    ax.tick_params(labelsize=12)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='12') 
    plt.show()

    #df1 = df[df['spi'] == 'MSR']
    ax = sns.displot(data=df, x="r2",hue='Type',col='Spectral Index', col_wrap=2,palette="Spectral",height=4,bins=10)

    a = ax.axes[0]
    a.axvline(0.4, color='red', ls='--', lw=3)

    b = ax.axes[1]
    b.axvline(0.4, color='red', ls='--', lw=3)

    c = ax.axes[2]
    c.axvline(0.4, color='red', ls='--', lw=3)
    ax.set(xlabel="$R^2$")
    plt.show()
