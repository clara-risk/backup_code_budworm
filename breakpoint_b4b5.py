#coding: utf-8

"""
Summary
-------
Breakpoint detection for automatic detection of outbreak period.

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
import fiona
import statsmodels.api as sm

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib
from osgeo import ogr, gdal,osr
from math import floor

import warnings
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks

def segment_fit(df,sp_idx):
    lonlat_tracker = [] 
    df_vals = df #.dropna(how='any')
    deriv = []
    year_min = []
    year_left = []
    year_right = []
    
    dataframes = []
    count1 = 0 
    for lon,lat,niveau in zip(list(df_vals['lon']),list(df_vals['lat']),list(df_vals['type'])):
        if (lon,lat,) not in lonlat_tracker:
            count1 +=1
            lonlat_tracker.append((lon,lat,))
            #time1= time.time()
            if count1 == 2:
                print('2')
            if count1 == 1000:
                print('1000')
            if count1 == 5000:
                print('5000') 
            if count1 == 10000:
                print('10,000')

            if count1 == 15000:
                print('15,000')

            if count1 == 16000:
                print('16,000')
                
            if count1 == 17000:
                print('17,000')

            if count1 == 20000:
                print('20,000')

            if count1 == 30000:
                print('30,000')

            if count1 == 40000:
                print('40,000')

            if count1 == 50000:
                print('50,000')

            if count1 == 100000:
                print('100,000')

            if count1 == 200000:
                print('200,000')

            if count1 == 300000:
                print('300,000')

            if count1 == 400000:
                print('400,000')

            if count1 == 410000:
                print('410,000')

            if count1 == 420000:
                print('420,000')

            if count1 == 500000:
                print('500,000')

            if count1 == 1000000:
                print('1,000,000')

            if count1 == 1100000:
                print('1,100,000')

            if count1 == 1500000:
                print('1,500,000')

            if count1 == 2000000:
                print('2,000,000')

            if count1 == 3000000:
                print('3,000,000')

            if count1 == 4000000:
                print('4,000,000')

            if count1 == 4500000:
                print('4,500,000')

            if count1 == 5000000:
                print('5,000,000')
            df_new = df_vals[df_vals['lon'] == lon]
            #print(df_new)
            #if lat > 48:
                #print('Point!')
            df_new = df_new[df_new['lat'] == lat] 
            df_new = df_new[['lon','lat','year',sp_idx,'type']].sort_values('year').dropna(how='any')
            #print(df_new)
            #lon_list.append(df_vals['lon'])
            #lat_list.append(df_vals['lat'])
            if len(df_new) >= 8: #2000
                #print(df_new)
                #df_affected = df_new[df_new['niveau'] == 1]
                #df_else = df_new[df_new['niveau'] == 0]
                #plt.scatter(df_else['year'],df_else[sp_idx],c='k',s=7,label='Spectral Index Value')
                #plt.scatter(df_affected['year'],df_affected[sp_idx],c='r',s=7,label='SBW Activity Detected')
                #plt.scatter(df_new['year'],df_new[sp_idx],c='k',s=7,label='Spectral Index Value')
                
                #trend = np.polyfit(df_new['year'],df_new[sp_idx],10)
                #trendpoly = np.poly1d(trend)
                #plt.plot(df_new['year'],trendpoly(df_new['year']),c='#229954')
                x = np.array(df_new['year'])
                y = np.array(df_new[sp_idx])
                peaks, _ = find_peaks(y,distance=5)
                ny = [ -x for x in y]
                valleys, _ = find_peaks(ny,distance=5)
##                if y[0] > 0:
##                    plt.plot(x, y,c='k')
##                    plt.scatter(x[peaks], y[peaks],facecolors='none',edgecolors='#D35400',s=50,label='Local Maximum')
##                    plt.scatter(x[valleys], y[valleys],facecolors='none',edgecolors='#2471A3',s=50,label='Local Minimum')
##                    plt.legend(frameon=False)
##                    plt.xlabel('Year')
##                    plt.ylabel(sp_idx+' Value')
##                    plt.show()
                    

                #Find where there is a max followed by a min 
                #print(x[peaks])
                #print(x[valleys])

                corresponding_peaks_left = {} 

                for valley in x[valleys]:
                    
                    peaks_list = x[peaks]
                    peak_filter_left = [] 
                    for peak in peaks_list:
                        if peak < valley: #cannot be equal, must be to left
                            peak_filter_left.append(peak)
                        
                    #Euclidean distance to all the maximums to the left of it
                    if len(peak_filter_left) > 0: 
                        distances = []
                        years = [] 
                        for peak in peak_filter_left:
                            time_d = valley-peak
                            distances.append(time_d)
                            years.append(peak)

                        closest_dist = distances.index(min(distances))
                        closest_left_peak = years[closest_dist]
                        corresponding_peaks_left[valley] = closest_left_peak
                    else:
                        pass

                    #print(corresponding_peaks_left)

                corresponding_peaks_right = {} 

                for valley in x[valleys]:
                    
                    peaks_list = x[peaks]
                    peak_filter_right = [] 
                    for peak in peaks_list:
                        if peak > valley: #cannot be equal, must be to right
                            peak_filter_right.append(peak)
                        
                    #Euclidean distance to all the maximums to the left of it
                    if len(peak_filter_right) > 0: 
                        distances = []
                        years = [] 
                        for peak in peak_filter_right:
                            time_d = peak - valley
                            distances.append(time_d)
                            years.append(peak)

                        closest_dist = distances.index(min(distances))
                        closest_right_peak = years[closest_dist]
                        corresponding_peaks_right[valley] = closest_right_peak
                    else:
                        corresponding_peaks_right[valley] = int( max(list(df_new['year'])) )

                    #print(corresponding_peaks_right)


                # Now merge the dictionaries based on the keys
                trendpoly_list = []
                years_broken = []
                second_deriv = []
                year_minimum = []
                for key, val in corresponding_peaks_left.items():
                    for key2, val2 in corresponding_peaks_right.items():

                        if key == key2:

                            #print(key)

                            year1 = val
                            #print(year1)
                            min_year = key
                            year2 = val2
                            #print(year2)

                            year1_idx = list(df_new['year']).index(year1)-1 #Add a padding to make sure it is fit correctly and to account for uncertainty in start year
                            if list(df_new['year']).index(year2) < max(list(df_new['year'])):
                                
                                year2_idx = list(df_new['year']).index(year2)+1
                            else:
                                year2_idx = list(df_new['year']).index(year2)

                            years = list(df_new['year'][year1_idx:year2_idx])

                            sp_vals = list(df_new[sp_idx][year1_idx:year2_idx])

                            if len(years) > 0:
                                
                                trend = np.polyfit(years,sp_vals,2, full=True)
                                trendpoly = np.poly1d(np.polyfit(years,sp_vals,2))

                    
                                SSE = trend[1][0]
                                diff = df_new[sp_idx] - np.nanmean(np.array(df_new[sp_idx]))
                                square_diff = diff ** 2
                                SST = square_diff.sum()
                                R2 = 1 - SSE/SST


                                if R2 >= 0.4:

                                    if trendpoly[0] > 0: #check for concave up 

                                        trendpoly_list.append(trendpoly)
                                        years_broken.append(years)
                                        #get year_min
                                        year_minimum.append(years[sp_vals.index(min(sp_vals))])
                                        delta = trendpoly.deriv(2)[0]
                                        second_deriv.append(delta)

##                                    else:
##                                        trendpoly_list.append(['D'])
##                                        years_broken.append(years)
##                                        second_deriv.append('D')
##                                        year_minimum.append('D')
##
##
##                                else:
##                                    trendpoly_list.append(['C'])
##                                    years_broken.append(years)
##                                    second_deriv.append('C')
##                                    year_minimum.append('C')
##                            else:
##                                trendpoly_list.append(['A'])
##                                years_broken.append(years)
##                                second_deriv.append('A')
##                                year_minimum.append('A')
                                
##
##                            else:
##                                deriv.append('SUP')
##                                year_min.append('SUP')
##                                year_left.append('SUP') #Remember we moved one over
##                                year_right.append('SUP')                                

                if list(df_new[sp_idx])[0] > 0: 
                    #plt.scatter(df_new['year'],df_new[sp_idx],c='k',s=7)
                    #plt.plot(df_new['year'],df_new[sp_idx])

                    #print(trendpoly_list)
                    count = 0 
                    for poly_inst,ylist in zip(trendpoly_list,years_broken):
                        if poly_inst[0] != 'SUP':
                            if count == 0: 
            
                                #plt.plot(ylist,poly_inst(ylist),c='r',label='Fitted Quadratic')
                                count +=1 

                            else:
                                #plt.plot(ylist,poly_inst(ylist),c='r')
                                count +=1 

                    #plt.legend(frameon=False)
                    #plt.xlabel('Year')
                    #plt.ylabel(sp_idx+' Value')

                    #plt.show()

                #Collect the following information
                #Years of detected outbreak
                #2nd derivative corresponding to those years
                #For other years, filter them out for manual classification

                #Find which ylist year is in

                df_list = [] 
                overall_count = len(year_minimum)
                count = 1
                for poly_inst,y_inst,deriv2,ym in zip(trendpoly_list,years_broken,second_deriv,year_minimum):
                    if count < overall_count: 
                    #df_new['deriv'] = []
                    #df_new['year_min'] = []
                    #df_new['year_left'] = []
                    #df_new['year_right'] = []

                        df_temp = df_new
                        
                        years_bt = list(range(y_inst[0],y_inst[-1]+1))
                        #years_bt = list(range(y_inst[0]+1,y_inst[-1]+1))
                        
                        #for y in range(y_inst[0],y_inst[-1]):
                        df_temp['d'+sp_idx] = [deriv2 if x in years_bt else np.nan for x in df_new['year']] #Cannot by 'SUP' here otherwise it will override earlier years
                        df_temp['ymin_'+sp_idx] = [ym if x in years_bt else np.nan for x in df_new['year']] 
                        df_temp['yleft_'+sp_idx] = [y_inst[0] if x in years_bt else np.nan for x in df_new['year']]
                        df_temp['yright_'+sp_idx] = [y_inst[-1] if x in years_bt else np.nan for x in df_new['year']]
                        df_temp = df_temp.dropna(how='any')
                        df_list.append(df_temp)
                        count+=1
                    else:

                        df_temp = df_new
                        
                        years_already_classed = []
                        for df in df_list:
                            years_already_classed.append(set(list(df['year'])))
                        
                        years_bt = list(range(y_inst[0],y_inst[-1]+1))
                        #years_bt = list(range(y_inst[0]+1,y_inst[-1]+1))
                        #years_already_classed.append(years_bt) #all years with records 
                        #for y in range(y_inst[0],y_inst[-1]):
                        df_temp['d'+sp_idx] = [deriv2 if x in years_bt else np.nan for x in df_new['year']] #Cannot by 'SUP' here otherwise it will override earlier years
                        df_temp['ymin_'+sp_idx] = [ym if x in years_bt else np.nan for x in df_new['year']] 
                        df_temp['yleft_'+sp_idx] = [y_inst[0] if x in years_bt else np.nan for x in df_new['year']]
                        df_temp['yright_'+sp_idx] = [y_inst[-1] if x in years_bt else np.nan for x in df_new['year']]
                        df_temp = df_temp.dropna(how='any')
                        years_already_classed.append(list(df_temp['year']))
                        concat_years = [j for i in years_already_classed for j in i]

                        noclass_years_info = []

                        concat_inverse = list(set(list(df_new['year'])) - set(concat_years))
                        #print(concat_inverse)
                        
                        for year_noclass in concat_inverse:
                            df_yr = df_new[df_new['year'] == year_noclass]
                            df_yr['d'+sp_idx] = 'SUP'
                            df_yr['ymin_'+sp_idx] = 'SUP'
                            df_yr['yleft_'+sp_idx] = 'SUP'
                            df_yr['yright_'+sp_idx] = 'SUP'                           
                            noclass_years_info.append(df_yr)

                        noclass_years_info.append(df_temp)
                        df_all_in = pd.concat(noclass_years_info)
                        #print(df_all_in)
                        df_list.append(df_all_in)
                        count+=1
                    

                if len(df_list) > 0:
                    #print('checkpoint3!')
                    df2 = pd.concat(df_list)
                    key_n = []
                    
                    for lon_n, lat_n, year_n in zip(df_new['lon'],df_new['lat'],df_new['year']):
                        key_n.append((lon_n,lat_n,year_n,))

                    key2 = [] 
                    for lon2, lat2, year2 in zip(df2['lon'],df2['lat'],df2['year']): 
                        key2.append((lon2,lat2,year2,))

                    for inst in key_n: 
                        if inst not in key2:
                            df_row = df_new.loc[(df_new['lon'] == inst[0]) & (df_new['lat'] == inst[1]) & \
                                                (df_new['year'] == inst[2])]
                            df2.append(df_row)
                    
                        
                    #df3 = df2
                            
                    df3 = pd.concat(df_list)
                    df3['combined'] = [(x,y,z) for x,y,z in zip(df3['lon'],df3['lat'],df3['year'])]
                    df3 = df3.drop_duplicates(subset='combined', keep="last") #Assign to LATER outbreak
                    
                    #df3 = df3.fillna('SUP').sort_values('year')
                    #df3 = df_list[-1]
                    if len(df3) != len(df_new):
                        print('Error!')
                        #sys.exit()
                        
                        print(df_new[['year',sp_idx,'d'+sp_idx,'ymin_'+sp_idx,'yleft_'+sp_idx,'yright_'+sp_idx]])
                        print(df3[['year',sp_idx,'d'+sp_idx,'ymin_'+sp_idx,'yleft_'+sp_idx,'yright_'+sp_idx]])
                    df_new = df3
                else:
                    #print('Checkpoint!') 
                    df_new['d'+sp_idx] = 'SUP'
                    df_new['ymin_'+sp_idx] = 'SUP'
                    df_new['yleft_'+sp_idx] = 'SUP'
                    df_new['yright_'+sp_idx] = 'SUP'   
                    #df_new.to_csv('Example.txt')
                    
                #if len(second_deriv) > 4 and second_deriv[0] < 0.03:
                    #print('Time series analysis in progress........................................') 
                    #print(second_deriv)
                    #print(df_new.head(50)[['lon','lat','year','deriv','year_min','year_left','year_right']])
##                    else: #I think this is appending a double value
##                        deriv.append(np.nan)
##                        year_min.append(np.nan)
##                        year_left.append(np.nan) #Remember we moved one over
##                        year_right.append(np.nan)                
                
                    
            else:
                #print('Checkpoint2!')
                
                df_new['d'+sp_idx] = np.nan
                df_new['ymin_'+sp_idx] = np.nan
                df_new['yleft_'+sp_idx] = np.nan
                df_new['yright_'+sp_idx] = np.nan

        else:
            pass

            
            #time2 = time.time()

            #elapsed = time2-time1
            #print('time:%s'%(elapsed))

        
        
        
        dataframes.append(df_new)
        
    n_dataframes = pd.concat(dataframes)
    #print(n_dataframes) 
    return n_dataframes



if __name__ == "__main__":

    path = 'outputs/b4b5/'

    years_df = []
    years_df_pt2 = []
    years_df_pt3 = []

    count = 0

    dfa = pd.read_csv('outputs/info/on_age.txt', delimiter = ",")
    dfa = dfa.sort_values('lon')
    dfa = dfa.sort_values('lat')
    list_age = dfa['age']

    dfb = pd.read_csv('outputs/info/on_bf.txt', delimiter = ",")
    dfb = dfb.sort_values('lon')
    dfb = dfb.sort_values('lat')
    list_bf = dfb['balsam_fir']

    dfb = pd.read_csv('outputs/info/on_bf.txt', delimiter = ",")
    dfb = dfb.sort_values('lon')
    dfb = dfb.sort_values('lat')
    list_bf = dfb['balsam_fir']

    dfw = pd.read_csv('outputs/info/on_ws.txt', delimiter = ",")
    dfw = dfw.sort_values('lon')
    dfw = dfw.sort_values('lat')
    list_ws = dfw['white_spruce']

    dfs = pd.read_csv('outputs/info/on_bs.txt', delimiter = ",")
    dfs = dfs.sort_values('lon')
    dfs = dfs.sort_values('lat')
    list_bs = dfs['black_spruce']
    
    
    for file in os.listdir(path):
        lookup_year = [str(x) for x in list(range(2014,2021+1))]
        if file.endswith('.txt') and any(n in file for n in lookup_year): 
            print('Reading %s.....................................................'%(file))
            if count ==0: 
                year = file[3:7]

                df = pd.read_csv(path+file, delimiter = ",")
                
                df = df.sort_values('lon')
                df = df.sort_values('lat')
                #df = df.sort_values('year')
                diff = float(year)-2011
                df['age'] = list_age
                df['age_upd'] = df['age'] + diff
                df['balsam_fir'] = list_bf
                df['white_spruce'] = list_ws
                df['black_spruce'] = list_bs
                df['combo'] = list_bf + list_ws + list_bs 
                #plt.scatter(df['lon'],df['lat'],s=0.25,c='orange')
                #plt.show()
                df = df.iloc[::100, :]

                
                #df = df.head(5000)
                #print(df)
                #plt.scatter(df['lon'],df['lat'],s=0.25,c=df['ndmi'])
                #plt.show()            
                #df = df.iloc[0:int(len(df)*0.2)]
                #df_niveau = pd.read_csv('outputs/on_'+str(year)+'.txt',delimiter = ',',names=['lon','lat','niveau'])
                #df2 = df_niveau[df_niveau['niveau'] == 1]
                #plt.scatter(df2['lon'],df2['lat'],c=df2['niveau'],s=0.2)
                #plt.show()
                length = len(df)
                print(length)
                #df['lat'] = df.lat.round(2).astype(float)
                #df['lon'] = df.lon.round(2).astype(float)
                df = df[df['age_upd'] > 0]
                df = df[df['combo'] > 0]
                print(len(df))
                years_df.append(df[['lon','lat','b4b5','year','type']])

                

                
                
            #years_df.append(df.iloc[0:int(len(df)*0.1)])
            #years_df_pt2.append(df.iloc[int(len(df)*0.1):int(len(df)*0.2)])
            #years_df_pt3.append(df.iloc[int(len(df)*0.2):int(len(df)*0.3)])
        
    df = pd.concat(years_df)
    
    #df2 = pd.concat(years_df_pt2)
    #df3 = pd.concat(years_df_pt3) 

    #df = df.head(100)

    #print(df) 
    #deriv, year_min_ndmi, year_left_ndmi, year_right_ndmi = segment_fit(df,'ndmi')
    
    #split dataframe and then multiprocess getting the derivative, then merge the lists
    
    start = time.time()
    df = segment_fit(df,'b4b5')
    df.to_csv('segmentation_2deriv_b4b5_2022_11_16_try.txt',mode='a',sep=',')
    #print('Completed chunk 1') 
##    df2 = segment_fit(df2,'nbr1')
##    df2.to_csv('segmentation_2deriv_2021_12_14.txt',mode='a',sep=',')
##    print('Completed chunk 2')
##
##    df3 = segment_fit(df3,'nbr1')
##    df3.to_csv('segmentation_2deriv_2021_12_14.txt',mode='a',sep=',')
##    print('Completed chunk 3') 
                            
    end = time.time()
    print('total time: '+str(end-start)) 
    t = end-start
    t_time = (length*t)/len(df)
    in_min = t_time/60
    in_hr = in_min/60
    print('Total est time in hr:'+str(in_hr))
    #deriv_msr, year_min_msr, year_left_msr, year_right_msr = segment_fit(df,'msr')
    #deriv_b4b5, year_min_b4b5, year_left_b4b5, year_right_b4b5 = segment_fit(df,'b4b5')

    #df.to_csv('segmentation_2deriv_2021_12_14.txt',sep=',')

            
    
