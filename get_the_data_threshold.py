     

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


if __name__ == "__main__":

    #Notes: pull out the affected so it's faster. Then pull out the buffered points. Write to csv.

    import random
    randomlist = random.sample(range(0, 7000), 7000)
    print(randomlist)

    from_crs = CRS('epsg:4326')

    years_df = {}

    new_df = pd.DataFrame(columns=['lat','lon'])
    count = 0
    storage = [] 
    for sp_index in ['b4b5','msr','nbr1','ndmi']: #'nbr1','ndmi',
        path = pathlib.Path(__file__).parent.resolve()
        df_idx = [] 
        for file in os.listdir(str(path)+'/outputs/'+sp_index):
            lookup_year = [str(x) for x in list(range(2014,2021+1))]
            
            if file.endswith('.txt') and sp_index in file and any(n in file for n in lookup_year): 
                print('Reading %s.....................................................'%(file))
                if count >= 0: 
                    year = file[3:7]

                    #n = 1000 #10% sample
                    #skip_func = lambda x: x%n != 0
                    #,skiprows = skip_func
                    df = pd.read_csv(str(path)+'/outputs/'+sp_index+'/'+file, delimiter = ",").dropna() #,skiprows = skip_func
                    print(df)
                    #df = df[df['type'] >= 1]
                    #df = pd.to_csv('track_affected_'+sp_index+'_'+year+'.txt', sep=',')
                    #df = pd.read_csv('track_affected_'+sp_index+'_'+year+'.txt', sep=',')
                    #print(df)
                    #new_df = new_df.append(df[['lat','lon']])
                    #print(new_df)
                    
                    df['lon'] = df['lon'].astype('float')
                    df = df.sort_values('lon')
                    df['lat'] = df['lat'].astype('float')
                    df = df.sort_values('lat')
                    df['year'] = df['year'].astype('float')
                    df = df.sort_values('year') 
                    length = len(df)
                    print(length)

                    
        
                    #df = df[::150]
                    

                    df_idx.append(df[['lon','lat','year','type',sp_index]])

                    #flight_lines1 = gpd.read_file('data/'+str(year)+'_ON_fixed.shp')
                    #flight_lines = flight_lines1.geometry.unary_union
                    #df_p = df[df['type'] >= 1] #or just >?? 
                    #df_p.to_csv(sp_index+'_'+year+'_affected.txt', sep=',')
                    #point = gpd.GeoDataFrame(df_p,geometry=gpd.points_from_xy(df_p['lon'],df_p['lat']))
                    #print('Computing intersection!') 
                    #df1 = sjoin(point, flight_lines1,op='within')
                    #print(df1)
                    #del df1['index_right']
                    
                    #df2 = out_of_area(df,year)
                    df_a = pd.read_csv(str(path)+'/track_affected.txt', delimiter = ",").dropna()[['lon','lat']]
                    
                    df_a = df_a.sort_values('lon')
                    
                    df_a = df_a.sort_values('lat')
                    
                    #df_a = df_a.sort_values('year') 

                    list_info = [[x,y] for x,y in zip(df_a['lat'],df_a['lon'])]
                    check_list = [[x,y] for x,y in zip(df['lat'],df['lon'])]
                    
                    df['latlon'] = [str([x,y]) for x, y in zip(df['lat'],df['lon'])]
                    df_a['latlon'] = [str([x,y]) for x, y in zip(df_a['lat'],df_a['lon'])]
                    print('checking if lat lon match') 
                    #df3 = df[df['latlon'].isin(list_info)]
                    df3 = pd.merge(df_a, df,  how='inner',on=['latlon']) #.reset_index(), left_on=['latlon'],right_on=['latlon']
                    print(df3)
                    print(df3[df3['type'] > 0])
                    df3['year'] = df3['year']
                    df3['type'] = df3['type']
                    df3['lon'] = df3['lon_y']
                    df3['lat'] = df3['lat_y']
                    df3 = df3[['lon','lat',sp_index,'year','type']].dropna()
                    print('complete')
                    print('concat') 
                    #df_p = pd.concat([df3,df_a])
                    print('complete') 
                    df3.to_csv(sp_index+'_'+year+'_affected_all.txt', sep=',')
                    #df2 = df2.sample(n=len(df_p)) #same amount as the affected
##                    df_p = pd.read_csv(str(path)+'/'+sp_index+'_'+year+'_affected_all.txt', delimiter = ",") #.dropna()
##                    
##                    df_p = df_p.sort_values('lon')
##                    
##                    df_p = df_p.sort_values('lat')
##                    
##                    df_p = df_p.sort_values('year')
##                    print(df_p)
##                    length = len(df)
                    

##                    g = df_p.groupby(['lon', 'lat'], sort=False)
##                    #print(g)
##                    #a=np.arange(g.ngroups)
##                    #a=np.arange(g.ngroups)
##
##                    #print(a)
##                    #a = np.random.shuffle(a)
##                    #print(a)
##
##                    df_p = df_p[g.ngroup().isin(randomlist)] #[:1500]
##                    
##                    print(df_p)
##                    df_train = pd.concat([df_p,df2]) #df
##                    
##                   
##                    df_idx.append(df_train[['lon','lat','year','type',sp_index]])
                    
##                    if int(year) >= 2014: 
##                        fig, ax = plt.subplots(figsize=(15, 15))
##                        cmap = gpd.read_file('data/on_cutout.shp')
##                        cmap.plot(ax=ax,facecolor='None',edgecolor='k')
##                        flight_lines1.plot(ax=ax,facecolor='None',edgecolor='k')
##                        
##                        plt.scatter(df1['lon'].astype(np.float16),df1['lat'].astype(np.float16) ,c='#E67E22',label='Positive Observation',s=1,alpha=0.75)
##                        plt.scatter(df2['lon'].astype(np.float16),df2['lat'].astype(np.float16) ,c='#1F618D',label='Negative Observation',s=1,alpha=0.75)
##                        plt.legend()
##                        plt.xlabel('Longitude')
##                        plt.ylabel('Latitude')  
##                        plt.show()
                    #df_idx.append(df[['lon','lat',sp_index,'year','type']])
                    count +=1 

        #yconcat = pd.concat(df_idx, ignore_index=True)
        
        #yconcat['combined'] = [(x,y,z,) for x,y,z in zip(yconcat['lon'],yconcat['lat'],yconcat['year'])]
        #years_df[sp_index] = yconcat #.set_index(['combined','lat','lon','year','type'])
    #new_df.to_csv('track_affected.txt',sep=',')
    #print(years_df)
    df_ndmi = years_df['ndmi'].drop_duplicates(keep='first') #.sample(frac=0.1, replace=True, random_state=1)
    
    df_msr = years_df['msr'].drop_duplicates(keep='first') #.sample(frac=0.1, replace=True, random_state=1)
    df_nbr1 = years_df['nbr1'].drop_duplicates(keep='first') #.sample(frac=0.1, replace=True, random_state=1)
    
    df_b4b5 = years_df['b4b5'].drop_duplicates(keep='first') #.sample(frac=0.1, replace=True, random_state=1)

    df_counter = df_ndmi[df_ndmi['year'] == 2020]

    R2_affected = []
    R2_unaffected = []

    R2_affected_msr = []
    R2_unaffected_msr = []

    R2_affected_nbr1 = []
    R2_unaffected_nbr1 = []

    R2_affected_b4b5 = []
    R2_unaffected_b4b5 = [] 
    
    start = time.time()
    count = 0
    
    lonlat_tracker = []
    
    for lon,lat in zip(list(df_counter['lon']),list(df_counter['lat'])):
        count+=1
        if (lon,lat,) not in lonlat_tracker:
            lonlat_tracker.append((lon,lat,))
            df_n = df_ndmi[df_ndmi['lon'] == lon]
            df_n = df_n[df_n['lat'] == lat]
            df_n = df_n.sort_values('year')

            df_m = df_msr[df_msr['lon'] == lon]
            df_m = df_m[df_m['lat'] == lat]
            df_m = df_m.sort_values('year')

            df_nb = df_nbr1[df_nbr1['lon'] == lon]
            df_nb = df_nb[df_nb['lat'] == lat]
            df_nb = df_nb.sort_values('year')

            df_b = df_b4b5[df_b4b5['lon'] == lon]
            df_b = df_b[df_b['lat'] == lat]
            df_b = df_b.sort_values('year')
            

            ndmi= list(df_n['ndmi'])
            msr= list(df_m['msr'])
            nbr1= list(df_nb['nbr1'])
            #print(nbr1)
            b4b5= list(df_b['b4b5'])
            #print(b4b5)

            if len(nbr1) == len(msr) == len(ndmi) == len(b4b5): 

                df_new = pd.DataFrame()
                df_new['lon'] = list(df_n['lon'])
                df_new['lat'] = list(df_n['lat'])
                df_new['ndmi'] = ndmi
                df_new['nbr1'] = nbr1
                df_new['msr'] = msr
                df_new['b4b5'] = b4b5
                df_new['year'] = list(df_n['year'])
                df_new['type'] = list(df_n['type'])
                df_new = df_new.sort_values('year')


                if len(df_new) >= 8:
                    #print(count)
                    #print(count/len(df_counter) *100)
                    #print(df_new)
                    delta = np.poly1d(np.polyfit(df_new.year, df_new.ndmi, 2)).deriv(2)[0]
                    values = list(df_new.ndmi)
                    index_min = min(range(len(values)), key=values.__getitem__)
                    min_year = list(df_new.year)[index_min]
                    fit = np.polyfit(df_new.year, df_new.ndmi, 2, full=True)
                    SSE = fit[1][0]
                    diff = df_new.ndmi - np.nanmean(np.array(df_new.ndmi))
                    square_diff = diff ** 2
                    SST = square_diff.sum()
                    R2 = 1 - SSE/SST
                    #print(R2)
                    #print(df_new['type'])
                    #if int(1) in list(df_new['type']) or int(2) in list(df_new['type']) or int(3) in list(df_new['type']):  #At least 1 year coded as affected
                    if any(i in list(df_new['type']) for i in [1,2,3]):
                        #print(df_new)
                        print('check')
                        print(R2_affected)
                        R2_affected.append(R2)
                    else:
                        R2_unaffected.append(R2)
                        
                    if count > 0 and int(2) in list(df_new['type']) and R2 >= 1.0 and delta > 0:
                        fig, ax = plt.subplots()
                        xs = df_new.year
                        ys = df_new.ndmi
                        zs = df_new.type

                        #print(xs)
                        #print(ys)
                        #print(zs)
                        
                        #mortality
                        ymort = [y for x,y,z in zip(xs,ys,list(df_new['type'])) if z == 2]
                        xmort = [x for x,y,z in zip(xs,ys,list(df_new['type'])) if z == 2]

                        trend = np.polyfit(xs,ys,2)
                        plt.scatter(xs,ys,c='k')
                        plt.scatter(xmort,ymort,c='b')
                        trendpoly = np.poly1d(trend) 
                        plt.plot(xs,trendpoly(xs),c='r')
                        
                        #plt.ylim([-1,1])
                        plt.xlabel('Year')
                        plt.ylabel('NDMI')
                        plt.text(0.8, 0.9,'2nd Derivative:'+str(round(delta,4)),horizontalalignment='center',verticalalignment='top',transform = ax.transAxes)
                        plt.text(0.8, 0.8,'R2:'+str(round(R2,4)),horizontalalignment='center',verticalalignment='top',transform = ax.transAxes)
            
                        plt.show()
                    fit_msr = np.polyfit(df_new.year, df_new.msr, 2, full=True)
                    values = list(df_new.msr)
                    index_min1 = min(range(len(values)), key=values.__getitem__)
                    min_year = list(df_new.year)[index_min1]
                    
                    SSE = fit_msr[1][0]
                    diff = df_new.msr - np.nanmean(np.array(df_new.msr))
                    square_diff = diff ** 2
                    SST = square_diff.sum()
                    R2 = 1 - SSE/SST

                    #if int(1) in list(df_new['type']) or int(2) in list(df_new['type']) or int(3) in list(df_new['type']): #At least 1 year coded as affected
                    if any(i in list(df_new['type']) for i in [1,2,3]): 
                        R2_affected_msr.append(R2)
                    else:
                        R2_unaffected_msr.append(R2)
                        
                    
                    fit_nbr1 = np.polyfit(df_new.year, df_new.nbr1, 2, full=True)
                    values = list(df_new.nbr1)
                    index_min2 = min(range(len(values)), key=values.__getitem__)
                    min_year = list(df_new.year)[index_min2]
                    
                    SSE = fit_nbr1[1][0]
                    diff = df_new.nbr1 - np.nanmean(np.array(df_new.nbr1))
                    square_diff = diff ** 2
                    SST = square_diff.sum()
                    R2 = 1 - SSE/SST

                    #if int(1) in list(df_new['type']) or int(2) in list(df_new['type']) or int(3) in list(df_new['type']):  #At least 1 year coded as affected
                    if any(i in list(df_new['type']) for i in [1,2,3]): 
                        R2_affected_nbr1.append(R2)
                    else:
                        R2_unaffected_nbr1.append(R2)
                    
                    delta_b4b5 = np.poly1d(np.polyfit(df_new.year, df_new.b4b5, 2)).deriv(2)[0]
                    values = list(df_new.b4b5)
                    index_min3 = min(range(len(values)), key=values.__getitem__)
                    min_year = list(df_new.year)[index_min3]
                    
                    fit_b4b5 = np.polyfit(df_new.year, df_new.b4b5, 2, full=True)
                    SSE = fit_b4b5[1][0]
                    diff = df_new.b4b5 - np.nanmean(np.array(df_new.b4b5))
                    square_diff = diff ** 2
                    SST = square_diff.sum()
                    R2 = 1 - SSE/SST

                    #if int(1) in list(df_new['type']) or int(2) in list(df_new['type']) or int(3) in list(df_new['type']): #At least 1 year coded as affected
                    if any(i in list(df_new['type']) for i in [1,2,3]): 
                        R2_affected_b4b5.append(R2)
                    else:
                        R2_unaffected_b4b5.append(R2)
            
##    R2_affected = list(pd.read_csv('ndmi_affected.txt',sep=',')['ndmi'])
##    R2_unaffected = list(pd.read_csv('ndmi_unaffected.txt',sep=',')['ndmi'])
##
##    R2_affected_msr = list(pd.read_csv('msr_affected.txt',sep=',')['msr'])
##    R2_unaffected_msr = list(pd.read_csv('msr_unaffected.txt',sep=',')['msr'])
##
##    R2_affected_nbr1 = list(pd.read_csv('nbr1_affected.txt',sep=',')['nbr1'])
##    R2_unaffected_nbr1 = list(pd.read_csv('nbr1_unaffected.txt',sep=',')['nbr1'])
##
##    R2_affected_b4b5 = list(pd.read_csv('b4b5_affected.txt',sep=',')['b4b5'])
##    R2_unaffected_b4b5 = list(pd.read_csv('b4b5_unaffected.txt',sep=',')['b4b5'])
    fig, ax = plt.subplots(2,2,sharex=True, sharey=True)
    matplotlib.rcParams.update({'font.size': 12})
    ax2 = ax[0,0].twinx()
    
    ax[0,0].hist([R2_affected, R2_unaffected], 10, None, ec='red', fc='none', lw=1.5, histtype='step')
    n, bins, patches = ax[0,0].hist([R2_affected, R2_unaffected], 10, None, ec='b', fc='none', lw=1.5, histtype='step')
    ax[0,0].cla() #clear the axis
    lns1 = ax[0,0].hist(R2_affected, 10,(0,1,), ec='red', fc='none', lw=1.5, histtype='step')
    lns2 = ax2.hist(R2_unaffected, 10, (0,1,), ec='blue', fc='none', lw=1.5, histtype='step')

    #ax.axvline(np.nanmean(np.array(R2_affected)), color='k', linestyle='dashed', linewidth=1)
    #ax.axvline(np.nanmean(np.array(R2_unaffected)), color='k', linestyle='dashed', linewidth=1)
    ax[0,0].axvline(0.6, color='k', linestyle='dashed', linewidth=1,label='Intersection')
    ax[0,0].set_title("NDMI")
    ax[0,0].tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,        
    labelbottom=False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelright='off')
    ax2.set_yticklabels([])
    ax2.set_ylim(bottom=0)
    ax[0,0].set_ylim(bottom=0)
    #ax.hist(R2_affected, 10, None, ec='red', fc='none', lw=1.5, histtype='step', label='Affected')
    #ax.hist(R2_unaffected, 10, None, ec='green', fc='none', lw=1.5, histtype='step', label='Unaffected')
    # added these three lines

    #fig.legend(loc='upper right')
    #ax2.legend(loc='upper center',bbox_to_anchor=(0.6,0.5))

    #ax.set_ylabel("Count, Affected")
    #ax2.set_ylabel("Count, Unaffected")
    #ax.set_xlabel("Coefficient of Determination R2")
    fig.suptitle('Histogram of Time Series Fit (R2)', fontsize=15)

    #fig, ax = plt.subplots()
    #matplotlib.rcParams.update({'font.size': 12})
    ax3 = ax[1,0].twinx()
    ax[1,0].hist([R2_affected_msr, R2_unaffected_msr], 10, None, ec='red', fc='none', lw=1.5, histtype='step')
    n, bins, patches = ax[1,0].hist([R2_affected_msr, R2_unaffected_msr], 10, None, ec='b', fc='none', lw=1.5, histtype='step')
    ax[1,0].cla() #clear the axis
    lns1 = ax[1,0].hist(R2_affected_msr, 10, (0,1,), ec='red', fc='none', lw=1.5, histtype='step')
    lns2 = ax3.hist(R2_unaffected_msr, 10,(0,1,), ec='blue', fc='none', lw=1.5, histtype='step')

    #ax[1,0].axvline(0.6, color='k', linestyle='dashed', linewidth=1,label='Intersection')
    ax[1,0].set_title("MSR")

    ax[1,0].spines['left'].set_visible(True)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(labelright='off')
    ax3.set_yticklabels([])     
    ax[1,0].set_ylabel("                                                               Count, Affected")
    ax[1,0].set_xlabel("                                                                                                                           Coefficient of Determination R2")
    ax3.set_ylim(bottom=0)
    ax[1,0].set_ylim(bottom=0)

    ax4 = ax[0,1].twinx()
    ax[0,1].hist([R2_affected_nbr1, R2_unaffected_nbr1], 10, None, ec='red', fc='none', lw=1.5, histtype='step')
    n, bins, patches = ax[0,1].hist([R2_affected_nbr1, R2_unaffected_nbr1], 10, None, ec='b', fc='none', lw=1.5, histtype='step')
    ax[0,1].cla() #clear the axis
    lns1 = ax[0,1].hist(R2_affected_nbr1,10, (0,1,), ec='red', fc='none', lw=1.5, histtype='step',label='Affected')
    lns2 = ax4.hist(R2_unaffected_nbr1, 10, (0,1,), ec='blue', fc='none', lw=1.5, histtype='step', label='Unaffected')

    ax[0,1].axvline(0.6, color='k', linestyle='dashed', linewidth=1,label='Intersection')
    ax[0,1].set_title("NBR1")
    ax[0,1].spines['left'].set_visible(False)
    ax[0,1].spines['right'].set_visible(True)

    ax5 = ax[1,1].twinx()
    ax5.set_ylabel("                                                               Count, Unaffected")
    ax[1,1].hist([R2_affected_b4b5, R2_unaffected_b4b5], 10, None, ec='red', fc='none', lw=1.5, histtype='step')
    n, bins, patches = ax[1,1].hist([R2_affected_b4b5, R2_unaffected_b4b5], 10, None, ec='b', fc='none', lw=1.5, histtype='step')
    ax[1,1].cla() #clear the axis
    lns1 = ax[1,1].hist(R2_affected_b4b5, 10, (0,1,), ec='red', fc='none', lw=1.5, histtype='step')
    lns2 = ax5.hist(R2_unaffected_b4b5, 10, (0,1,), ec='blue', fc='none', lw=1.5, histtype='step')
    fig.legend(loc='upper right')
    ax[1,1].axvline(0.6, color='k', linestyle='dashed', linewidth=1,label='Intersection')
    ax[1,1].set_title("B4B5 Ratio")
    ax[1,1].spines['left'].set_visible(False)
    ax[1,1].spines['right'].set_visible(True)
    plt.show()

    df_winter_affected_n = pd.DataFrame()
    df_winter_affected_n['ndmi'] = R2_affected
    df_winter_affected_m = pd.DataFrame()
    df_winter_affected_m['msr'] = R2_affected_msr
    df_winter_affected_nb = pd.DataFrame()
    df_winter_affected_nb['nbr1'] = R2_affected_nbr1
    df_winter_affected_b = pd.DataFrame()
    df_winter_affected_b['b4b5'] = R2_affected_b4b5

    df_winter_unaffected_n = pd.DataFrame()
    df_winter_unaffected_n['ndmi'] = R2_unaffected
    df_winter_unaffected_m = pd.DataFrame()
    df_winter_unaffected_m['msr'] = R2_unaffected_msr
    df_winter_unaffected_nb = pd.DataFrame()
    df_winter_unaffected_nb['nbr1'] = R2_unaffected_nbr1
    df_winter_unaffected_b = pd.DataFrame()
    df_winter_unaffected_b['b4b5'] = R2_unaffected_b4b5

    df_list = [df_winter_affected_n,df_winter_affected_m,df_winter_affected_nb,df_winter_affected_b]
    names = ['ndmi_affected','msr_affected','nbr1_affected','b4b5_affected']
    for df,name in zip(df_list,names): 
        
        df.to_csv(name+'.txt', sep=',')

    df_list = [df_winter_unaffected_n,df_winter_unaffected_m,df_winter_unaffected_nb,df_winter_unaffected_b]
    names = ['ndmi_unaffected','msr_unaffected','nbr1_unaffected','b4b5_unaffected']
    for df,name in zip(df_list,names): 
        
        df.to_csv(name+'.txt', sep=',')

