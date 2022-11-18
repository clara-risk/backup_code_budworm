#coding: utf-8

"""
Summary
-------
Script to (A) produce sample points, (B) obtain values from rasters for covariates & spectral indices for 
every sample point, and (C) output information to a csv file. 

References 
----------
Calculating DEV ( deviation from mean elevation ): De Reu et al. (2013), see:
De Reu, J., J. Bourgeois, M. Bats, A. Zwertvaegher, V. Gelorini, P. De Smedt, W. Chu, M. Antrop, P. De Maeyer, 
P. Finke, M. Van Meirvenne, J. Verniers, and P. Crombé. 2013. Application of the topographic position index 
to heterogeneous landscapes. Geomorphology 186.
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
from descartes import PolygonPatch
from shapely.ops import cascaded_union, unary_union
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
from osgeo import ogr, gdal,osr
from math import floor

import warnings
warnings.filterwarnings('ignore')

def ini_points_ont(bounds,spacing, aerial_survey_shp,provincial_shp,filename):
    '''Create a lookup file of the sample points, including the presence/absence
    of defoliation/mortality information from the aerial survey data 
    Writes a text file to the outputs folder in the directory
    Parameters
    ----------
    bounds : list
        a list of the lat/lon bounds of the study area in the format 
        [min lon, max lon, min lat, max lat]
    spacing : int
        space between the sample points in meters, ex 10000 m for 10 km
    aerial_survey_shp : shapefile
        shapefile including the defoliation/mortality maps for year of interest
    provincial_shp : shapefile
        shapefile with provincial boundaries and large waterbodies, ex
        St Lawrence Seaway 
    filename : string 
        name of lookup file to be written to the drive
    '''
    na_map = gpd.read_file(provincial_shp)
    boundary = na_map.to_crs('EPSG:32618') #Ocean
    boundary = boundary['geometry'].unary_union
    boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary))
    
    aerial_survey = gpd.read_file(aerial_survey_shp)
    aerial_survey = aerial_survey.to_crs('EPSG:32618')

    xmax = bounds[1]
    xmin= bounds[0]
    ymax = bounds[3]
    ymin = bounds[2]
    
    from_crs = CRS('epsg:4326')
    transformer = Transformer.from_crs(from_crs,'EPSG:32618',always_xy=True)

    xmax,ymax = transformer.transform(xmax,ymax)
    xmin,ymin = transformer.transform(xmin,ymin)

    num_col = int((xmax - xmin) / spacing) 
    num_row = int((ymax - ymin) / spacing)

    Yi = np.linspace(ymin,ymax,num_row+1)
    Xi = np.linspace(xmin,xmax,num_col+1)
    Xi,Yi = np.meshgrid(Xi,Yi)

    points = [Point(x) for x in zip(Xi.flatten(),Yi.flatten())]
    print(len(points))
    points = gpd.GeoDataFrame(geometry=gpd.GeoSeries(points))
    points['lat'] = list(Yi.flatten())
    points['lon'] = list(Xi.flatten())
    points = sjoin(points, boundary, op = 'within')
    print(points.head(10))
    print(len(points))
    
    del points['index_right']
    
##    print('check if in study area 3')
##    fig,ax=plt.subplots(1,1)
##    ax.scatter(Xi.flatten(),Yi.flatten(),s=0.25)
##    boundary.plot(ax=ax,facecolor="none", edgecolor='k',linewidth=1,zorder=20)
##    #plt.show() 
##    #points_filter = boundary.contains(points)
##    points_filter = points.geometry.within(boundary)
##    print(points_filter)
##    points_filter = points[points_filter]
##    print(points_filter)
## 
    points_in_shp = sjoin(points, aerial_survey, how='left')
    print(list(points_in_shp))
   
    if 'FOREST_INS' in list(points_in_shp): 
##        del points_in_shp['YEAR_OF_EV']
##        del points_in_shp['FOREST_DAM']
##        del points_in_shp['DAMAGE_EVE']
##        del points_in_shp['LOCATION_A']
##        del points_in_shp['GEOMETRY_U']
##        del points_in_shp['EFFECTIVE_']
##        del points_in_shp['SYSTEM_CAL'] #_DAT
        #del points_in_shp['Shape_Leng']
        #del points_in_shp['Shape_Area']
        #del points_in_shp['index_right']

        points_in_shp['FOREST_INS'] = points_in_shp['FOREST_INS'].fillna(0)

        points_in_shp.loc[points_in_shp['FOREST_INS'] != 0, 'FOREST_INS'] = 1
        
        points_in_shp['NIVEAU'] = points_in_shp['FOREST_INS']
        
    elif 'D'+aerial_survey_shp[5:9] in list(points_in_shp):
        points_in_shp['D'+aerial_survey_shp[5:9]] = points_in_shp['D'+aerial_survey_shp[5:9]].fillna(0)

        points_in_shp.loc[points_in_shp['D'+aerial_survey_shp[5:9]] != 0, 'D'+aerial_survey_shp[5:9]] = 1

        points_in_shp['NIVEAU'] = points_in_shp['D'+aerial_survey_shp[5:9]]
        
    elif 'Level' in list(points_in_shp):
        points_in_shp['Level'] = points_in_shp['Level'].fillna(0)

        points_in_shp.loc[points_in_shp['Level'] != 0, 'Level'] = 1

        points_in_shp['NIVEAU'] = points_in_shp['Level']
    else:
        print(points_in_shp)
        
        points_in_shp['FID'] = points_in_shp['OGF_FID'].fillna(0) #OGF_ID for 2021 otherwise FID

        points_in_shp.loc[points_in_shp['FID'] != 0, 'FID'] = 1

        points_in_shp['NIVEAU'] = points_in_shp['FID'] 

    array = np.array(points_in_shp['NIVEAU'])
    Xi = np.array(points_in_shp['lon'])
    Yi = np.array(points_in_shp['lat'])
    from_crs = CRS('EPSG:32618')
    transformer = Transformer.from_crs(from_crs,'EPSG:4326',always_xy=True)
    Xi,Yi = transformer.transform(Xi,Yi) 

    new_lon = []
    new_lat = []
    new_niveau = [] 
    for niveau,lon,lat in zip(array.flatten(),Xi.flatten(),Yi.flatten()):
        stacker = np.dstack((niveau,lon,lat))
        new_lon.append(lon)
        new_lat.append(lat)
        new_niveau.append(niveau)

    new_data = pd.DataFrame() 
    new_data['lon'] = new_lon
    new_data['lat'] = new_lat
    new_data['niveau'] = new_niveau
    
    
    print('created df') 
            
    new_data.to_csv('outputs/'+filename+'.txt',index=False,sep=',')
    print('complete') 

##    with open('outputs/'+filename+'.txt', 'a') as f:
##        for sub_inst in zip(array.flatten(),Xi.flatten(),Yi.flatten()):
##            point = Point(sub_inst[1],sub_inst[2])
##            point = gpd.GeoDataFrame(geometry=gpd.GeoSeries(point)) 
##            if point.geometry.within(study_area).any(): 
##                f.write('{:f}'.format(sub_inst[0])+','+'{:f}'.format(sub_inst[1])+','+'{:f}'.format(sub_inst[2])+'\n')
##                #if count % 1000:
##                    #print('{:f}'.format(sub_inst[0])+','+'{:f}'.format(sub_inst[1])+','+'{:f}'.format(sub_inst[2]))
##                count+=1 

def ini_points_match(bounds,spacing,txt,aerial_survey_shp,provincial_shp,out_file):
    aerial_survey = gpd.read_file(aerial_survey_shp)

    df_pts = pd.read_csv(txt,delimiter=',')
    lon = df_pts['lon']
    lat = df_pts['lat']
    print(len(df_pts))
    
    points = [Point(x) for x in zip(lon,lat)]
    points = gpd.GeoDataFrame(df_pts[['lon','lat']],geometry=gpd.GeoSeries(points))

    print(len(points.drop_duplicates()))
    
    points_in_shp = sjoin(points.reset_index(drop=True), aerial_survey, how='left') #op = 'within')
    points_in_shp = points_in_shp.drop_duplicates(subset=['lon', 'lat'])
    print(points_in_shp)
   
    if 'FOREST_INS' in list(points_in_shp):
        print('FOREST_INS') 
##        del points_in_shp['YEAR_OF_EV']
##        del points_in_shp['FOREST_DAM']
##        del points_in_shp['DAMAGE_EVE']
##        del points_in_shp['LOCATION_A']
##        del points_in_shp['GEOMETRY_U']
##        del points_in_shp['EFFECTIVE_']
##        del points_in_shp['SYSTEM_CAL'] #_DAT
        #del points_in_shp['Shape_Leng']
        #del points_in_shp['Shape_Area']
        #del points_in_shp['index_right']

        points_in_shp['FOREST_INS'] = points_in_shp['FOREST_INS'].fillna(0)

        points_in_shp.loc[points_in_shp['FOREST_INS'] != 0, 'FOREST_INS'] = 1
        
        points_in_shp['NIVEAU'] = points_in_shp['FOREST_INS']
        
    elif 'D'+aerial_survey_shp[5:9] in list(points_in_shp):
        print('D') 
        points_in_shp['D'+aerial_survey_shp[5:9]] = points_in_shp['D'+aerial_survey_shp[5:9]].fillna(0)

        points_in_shp.loc[points_in_shp['D'+aerial_survey_shp[5:9]] != 0, 'D'+aerial_survey_shp[5:9]] = 1

        points_in_shp['NIVEAU'] = points_in_shp['D'+aerial_survey_shp[5:9]]
        
    elif 'Level' in list(points_in_shp):
        print('Level!') 
        points_in_shp['Level'] = points_in_shp['Level'].fillna(0)

        points_in_shp.loc[points_in_shp['Level'] != 0, 'Level'] = 1

        points_in_shp['NIVEAU'] = points_in_shp['Level']
    else:
        print('Check!')
        
        points_in_shp['FID'] = points_in_shp['OGF_ID'].fillna(0)

        points_in_shp.loc[points_in_shp['FID'] != 0, 'FID'] = 1

        points_in_shp['NIVEAU'] = points_in_shp['FID'] 

    array = np.array(points_in_shp['NIVEAU'])
    Xi = np.array(points_in_shp['lon'])
    Yi = np.array(points_in_shp['lat'])
    print(len(array))
    print(len(Xi)) 

    new_lon = []
    new_lat = []
    new_niveau = [] 
    for niveau,lon,lat in zip(array.flatten(),Xi.flatten(),Yi.flatten()):
        stacker = np.dstack((niveau,lon,lat))
        new_lon.append(lon)
        new_lat.append(lat)
        new_niveau.append(niveau)

    new_data = pd.DataFrame() 
    new_data['lon'] = new_lon
    new_data['lat'] = new_lat
    new_data['niveau'] = new_niveau
    
    
    print('created df') 
            
    new_data.to_csv(out_file+'.txt',index=False,sep=',')
    print('complete') 
                
def get_dataframe(year,df_merger,study_area,shp):
    '''Create the training and testing Pandas dataframes needed for training and testing the GAM and RF models
    Parameters
    ----------
    year : int
        year of interest
    df_merger : Pandas dataFrame
        pre-created dataframe for first/last year in outbreak to match the new dataframe to 
    study_area : string
        study area name, must match name used elsewhere
    shp : shapefile
        study area shapefile, with waterbodies delineated 
    Returns
    ----------
    Pandas dataframe
        - Dataframe containing the information from the raster for each point 
    '''
    df = get_val_frm_raster(year,study_area+'_'+str(year),study_area+'_ndmi_'+str(year),'ndmi','data/'+shp+'.shp')
    df_msr = get_val_frm_raster(year,study_area+'_'+str(year),study_area+'_msr_'+str(year),'msr','data/'+shp+'.shp')
    df_nbr1 = get_val_frm_raster(year,study_area+'_'+str(year),study_area+'_nbr1_'+str(year),'nbr1','data/'+shp+'.shp')
    df_b4b5 = get_val_frm_raster(year,study_area+'_'+str(year),study_area+'_b4b5ratio_'+str(year),'b4b5','data/'+shp+'.shp')
    
    df = df[['lon','lat','ndmi','ndmi_left','ndmi_right','ndmi_above','ndmi_below']].merge(df_msr, left_on=['lon', 'lat'], right_on=['lon', 'lat'], how='left', indicator=True)
    del df['_merge']
    df = df[['lon','lat','ndmi','ndmi_left','ndmi_right','ndmi_above','ndmi_below','msr',\
                     'msr_left','msr_right','msr_above','msr_below']].merge(df_nbr1, left_on=['lon', 'lat'], right_on=['lon', 'lat'], indicator=True)
    del df['_merge']
    df = df[['lon','lat','ndmi','ndmi_left','ndmi_right','ndmi_above','ndmi_below','msr',\
                     'msr_left','msr_right','msr_above','msr_below','nbr1',\
                     'nbr1_left','nbr1_right','nbr1_above','nbr1_below']].merge(df_b4b5, left_on=['lon', 'lat'], right_on=['lon', 'lat'], indicator=True)
    del df['_merge']
    
    return df

def get_val_frm_raster(year,file_name,file_name_raster,index_name,shp,dev=False,neighborhood_size=1):
    '''Lookup the value for the sample point (read from text file) for a specific raster
    Parameters
    ----------
    year : int
        year of interest
    file_name : string
        file name of the lookup file for the sample points, not including .txt ending
    file_name_raster : string
        file name of raster, not including the .tif ending 
    index_name : string 
        column name to use for raster value in the resulting dataframe
    shp : shapefile
        shapefile of the study area delineating boundary and waterbodies
    dev : bool
        whether or not you are calculating the mean deviation of elevation
    neighborhood_size : int
        number of pixels surrounding the original pixel to consider when calculating
        the DEV, 1 pixel buffer = 8 pixels total; 2 pixel buffer = 24 pixels total
    Returns
    ----------
    Pandas dataframe
        - Dataframe containing the information from the raster for each point
    '''
    #for file in (directory):  
    src_ds = gdal.Open('outputs/'+file_name_raster+'.tif')
    rb1=src_ds.GetRasterBand(1)
    transform=src_ds.GetGeoTransform()
    print('Success in reading file.........................................') 

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    df = pd.read_csv('outputs/'+file_name+'.txt', delimiter = ',')
    #df.columns = ['lon', 'lat','type']
    print(df)

    comb_niveau = df['type']
    comb_points = [Point(x) for x in zip(df['lon'],df['lat'])]

    #comb_points = comb_points[:3090] #for testing

    #na_map = gpd.read_file(shp)
    #gdf = gpd.GeoDataFrame(df,geometry=gpd.GeoSeries(comb_points))
    #intersection = gpd.sjoin(na_map,gdf,how='left',op='intersects')
    #gdf = intersection[intersection['niveau'].notna()]

    #new_niveau = gdf['niveau']
    #comb_points_inside = [Point(x) for x in zip(gdf['lon'],gdf['lat'])]
    new_niveau = comb_niveau
    comb_points_inside = comb_points


    x_vals = []
    y_vals = []
    z_vals = []
    left_vals = []
    right_vals = []
    above_vals = []
    below_vals = []
    niveau = []
    dev_list = [] 
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize

    #data = rb1.ReadAsArray(0, 0, cols, rows)
    data = rb1.ReadAsArray(0, 0, cols, rows)

    for feat,val in zip(comb_points_inside,new_niveau):
        geom = feat
        niveau1= val

        mx,my=np.array(geom.coords.xy[0])[0], np.array(geom.coords.xy[1])[0]  

        col = int((mx - xOrigin) / pixelWidth)
        row = int((yOrigin - my ) / pixelHeight)

        # Get the surrounding 4 pixels

        if dev == True:
            surroundings = []
            for shift in list(range(1,neighborhood_size+1)):
                col_left = col-shift
                col_right = col+shift
                row_above = row+shift
                row_below = row-shift
                if col > 0 and row >0 and col < cols-neighborhood_size and row < rows-neighborhood_size:
                    #Rook 
                    surroundings.append(data[row][col_left])
                    surroundings.append(data[row][col_right])
                    surroundings.append(data[row_above][col])
                    surroundings.append(data[row_below][col])
                    #Add Queen
                    surroundings.append(data[row_above][col_left])
                    surroundings.append(data[row_above][col_right])
                    surroundings.append(data[row_below][col_right])
                    surroundings.append(data[row_below][col_right])
                else:
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)
                    #Add Queen
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)
                    surroundings.append(np.nan)                    
            #melev = sum(surroundings) / len(surroundings)
            melev = np.nanmean(surroundings)
            diff = [(val-melev)**2 for val in surroundings if not np.isnan(val)]
            sd = math.sqrt( ( 1/(len(diff)-1) ) * sum(diff) )
            dev_val = ( data[row][col] - melev ) / sd 
            dev_list.append(dev_val)


        col_left = col-1
        col_right = col+1
        row_above = row+1
        row_below = row-1


        if col > 0 and row >0 and col < cols-1 and row < rows-1:

            x_vals.append(mx)
            y_vals.append(my)
            z_vals.append(data[row][col])
            #left_vals.append(data[row][col_left])
            #right_vals.append(data[row][col_right])
            #above_vals.append(data[row_above][col])
            #below_vals.append(data[row_below][col])
            niveau.append(niveau1)

        else:
            x_vals.append(mx)
            y_vals.append(my)
            z_vals.append(np.nan)
            #left_vals.append(np.nan)
            #right_vals.append(np.nan)
            #above_vals.append(np.nan)
            #below_vals.append(np.nan)
            niveau.append(niveau1)
    if dev == True:
                
        df_test = pd.DataFrame.from_records(zip(x_vals,y_vals,z_vals,niveau,dev_list), \
                                        columns=['lon','lat',index_name,'niveau','dev'])
        #print(df_test)

        df_new = df_test.assign(year=year)

    else:
        print('Creating dataframe............................................................') 
        df_test = pd.DataFrame.from_records(zip(x_vals,y_vals,z_vals,niveau), \
                                        columns=['lon','lat',index_name,'niveau'])

        df_new = df_test.assign(year=year)        

    return df_new 
    
    
if __name__ == "__main__":

    # Make the sample points lookup files 
    
    #years = [2002,2009,2011]#[2009,2008,2006,2005,2004,2003,2002,2001,2000] #[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020] #1999,1998,1997,1996,1995,1994,1993,1992,1991,1990,1989,1988,1987,
    #years = list(range(1984,1985))
    #for year in years:
        #print(year)
    
        #ini_points_ont([-98,-73, 39, 59],500,'data/'+str(year)+'_ON_fixed.shp','data/fixed_boreal_cutout.shp','on_'+str(year))
        #ini_points_match([-98,-73, 39, 57],500,'outputs/on_'+str(2018)+'.txt','data/'+str(year)+'_ON_fixed.shp','data/boreal_cutout.shp','on_'+str(year))
    years = list(range(2021,2021+1))
##    for year in years:
##        #print(year)
##    
##        ini_points_match([-98,-73, 39, 59],500,'outputs/on_'+str(1984)+'.txt',\
##                         'data/'+str(year)+'_ON_fixed.shp','data/fixed_boreal_cutout.shp','on_'+str(year))
        #ini_points_match([-98,-73, 39, 57],500,'outputs/on_'+str(2018)+'.txt','data/'+str(year)+'_ON_fixed.shp','data/boreal_cutout.shp','on_'+str(year))
 
    #year = 1988

    #df2014 = get_val_frm_raster(year,'on_'+str(year),'on_ndmi_'+str(year),'ndmi','data/on_cutout.shp')
    #df2014_msr = get_val_frm_raster(year,'on_'+str(year),'on_msr_'+str(year),'msr','data/on_cutout.shp')
    for year in years: 
        #df2014 = get_val_frm_raster(year,'on_'+str(year),'on_b4b5_500m_'+str(year),'b4b5','data/boreal_cutout.shp')
        df2014 = get_val_frm_raster(year,'on_'+str(year),'black_spruce','black_spruce','data/boreal_cutout.shp')
        
    #df2014_b4b5 = get_val_frm_raster(year,'on_'+str(year),'on_b4b5ratio_'+str(year),'b4b5','data/on_cutout.shp')
        #df2014['niveau'] = np.where(df2014['niveau']>=1,1,0)
        df2014.to_csv('outputs/info/on_'+str(year)+'_bs.txt',sep=',')
    
        print('Complete................................................') 
    time.sleep(60)
##    
##    df2015 = get_dataframe(2015,df2014,'nco','nco_cutout')
##    df2015['dev'] = list(df2014['dev'])
##    df2015['elevation'] = list(df2014['elevation'])
##    df2015['white_spruce'] = list(df2014['white_spruce'])
##    df2015['black_spruce'] = list(df2014['black_spruce'])
##    df2015['balsam_fir'] = list(df2014['balsam_fir'])
##    df2015['soil_text'] = list(df2014['soil_text'])
##    
##    if list(df2015['lon']) != list(df2014['lon']):
##        print('error')
##        
##    df2016 = get_dataframe(2016,df2014,'nco','nco_cutout')
##    df2016['dev'] = list(df2014['dev'])
##    df2016['elevation'] = list(df2014['elevation'])
##    df2016['white_spruce'] = list(df2014['white_spruce'])
##    df2016['black_spruce'] = list(df2014['black_spruce'])
##    df2016['balsam_fir'] = list(df2014['balsam_fir'])
##    df2016['soil_text'] = list(df2014['soil_text'])
##    
##    df2017 = get_dataframe(2017,df2014,'nco','nco_cutout')
##    df2017['dev'] = list(df2014['dev'])
##    df2017['elevation'] = list(df2014['elevation']) 
##    df2017['white_spruce'] = list(df2014['white_spruce']) 
##    df2017['black_spruce'] = list(df2014['black_spruce'])  
##    df2017['balsam_fir'] = list(df2014['balsam_fir'])
##    df2017['soil_text'] = list(df2014['soil_text'])
##    
##    df2018 = get_dataframe(2018,df2014,'nco','nco_cutout')
##    df2018['dev'] = list(df2014['dev'])
##    df2018['elevation'] = list(df2014['elevation'])
##    df2018['white_spruce'] = list(df2014['white_spruce'])
##    df2018['black_spruce'] = list(df2014['black_spruce'])
##    df2018['balsam_fir'] = list(df2014['balsam_fir'])
##    df2018['soil_text'] = list(df2014['soil_text'])
##
##    df2019 = get_dataframe(2019,df2014,'nco','nco_cutout')
##    df2019['dev'] = list(df2014['dev'])
##    df2019['elevation'] = list(df2014['elevation'])
##    df2019['white_spruce'] = list(df2014['white_spruce']) 
##    df2019['black_spruce'] = list(df2014['black_spruce'])
##    df2019['balsam_fir'] = list(df2014['balsam_fir'])
##    df2019['soil_text'] = list(df2014['soil_text'])
## 
##    df2020 = get_dataframe(2020,df2014,'nco','nco_cutout')
##    df2020['dev'] = list(df2014['dev'])
##    df2020['elevation'] = list(df2014['elevation'])
##    df2020['white_spruce'] = list(df2014['white_spruce'])
##    df2020['black_spruce'] = list(df2014['black_spruce'])
##    df2020['balsam_fir'] = list(df2014['balsam_fir'])
##    df2020['soil_text'] = list(df2014['soil_text'])
##    
##    df_list = [df2014,df2015,df2016,df2017,df2018,df2019,df2020]
##
##    years_list = ['2014','2015','2016','2017','2018','2019','2020']
##
##    df_dict = dict(zip(years_list, df_list))
##
##    for dtf in df_list:
##        dtf.reset_index(drop=True, inplace=True)
##    
##    merge_df = gpd.GeoDataFrame(pd.concat([df2014.reset_index(drop=True),df2015.reset_index(drop=True),df2016.reset_index(drop=True)\
##                                           ,df2017.reset_index(drop=True),df2018.reset_index(drop=True),df2019.reset_index(drop=True),df2020.reset_index(drop=True)], \
##                                          ignore_index=True))
##    
##    string_years = merge_df['year'].unique()
##    string_years = [str(x) for x in string_years]
##    
##    dfy = [] 
##
##    for year in string_years:
##
##        df_new1 = df_dict[str(year)]
##        print(df_new1)
##
##        flight_lines = gpd.read_file('data/union/'+year+'.shp')
##        flight_lines = flight_lines.geometry.unary_union
##        
##        point = gpd.GeoDataFrame(df_new1,geometry=gpd.points_from_xy(df_new1['lon'],df_new1['lat']))
##
##        df_new1 = point[point.geometry.within(flight_lines)]
##        
##        #df_new1 = sjoin(point, flight_lines, how='left',op='within')
##        print(df_new1)
##
##        #del df_new1['index_right']
##        #del df_new1['FID']
##
##        dfy.append(df_new1)
##
##    print('Finished filtering sample points based on flightlines & getting leading edge info') 
##    
##    df_new = pd.concat(dfy)   
##    
##    df_new.to_csv('data_in_flightlines.csv',sep=',')
##    
##    
##    dfy_val = [] 
##
##    for year in string_years: 
##        df_year = df_dict[str(year)]
##
##        dfy_val.append(df_year) 
##
##    df_vals = pd.concat(dfy_val)
##    df_vals.to_csv('data_overall.csv',sep=',')  
##    
