
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

def byhex(df,referent,vx,vy,vbound_lower,vbound_upper,var,p_low,p_up):

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
    plt.plot(x, y)
    plt.show()
    for hexcode in hexes:

        #print(hexcode)
        hdf = df[df['hex'] == hexcode] 
        if hexcode not in referent:
            val = float(hdf[var])
            print(val) 
            if val >= vbound_lower and val <= vbound_upper:
                odds = 1 # 1 or 0? 
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

    df = df[df['elev'] >= 261]
    df = df[df['elev'] <= 299]
    df = df[df['bf'] <= 2]
    df = df[df['sb'] >= 34]
    df = df[df['sb'] <= 45]
    df = df[df['mj'] <= -25.26]
    df = df[df['mj'] >= -26.37]
    df = df[df['age'] >= 70]
    df = df[df['age'] <= 79]

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

    print(len(df))

    print(df[df['hex'] == -1])
    
    df = df[df['hex'] != -1]

    print(len(set(list(df['hex']))))
    
    hex_filter = df.groupby(['hex'],dropna=False).mean().reset_index()

    print(hex_filter)

    asm = gpd.read_file('grid_5000.shp').to_crs('EPSG:4326')

    hex_list = asm['id']

    elev_range = [0, 20, 40, 60, 80, 100, 120, 140, 160,\
                180, 200, 220, 240, 260, 280, 300, 320, 340,\
                360, 380, 400, 420, 440, 460, 480, 500, 520, 540]
    elev_odds = [1.1237824125161196e-10, 1.2877791051882814e-119, 2.4318917628356207e-10,
                 2.351301996995713e-15, 3.6453688496165293e-10, 0.3901096868484866, 0.880188332821613,
                 1.2129130336105842, 1.1797945531305165, 0.8954568915079646, 0.7158946335770928,
                 1.5539479090666426, 1.7185455374648009, 1.7432549707418925, 1.2699217053124323,
                 1.6805501643285137, 1.727720063317128, 1.751242625444832,
                 1.8062645696965811, 2.275187844809393, 3.252428673852006,
                 4.387286578076808, 6.639516765992007, 9.785278649451428, 14.134687425947524,
                 17.84988886858476, 18.326446960641963, 17.522541295164373]
    
    

    elev = byhex(hex_filter,[],elev_range,elev_odds,261,299,'elev',0,500)

    brange = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    
    bodds = [1.0357943703772252, 1.051165234687968, 1.0889626021127263, 1.2539491113889731,
             1.6338608346573915, 1.9861522557952824, 2.117731790640132, 2.9880256350759278,
             4.959697738191339, 5.369129003758281]

    bf = byhex(hex_filter,[],brange,bodds,0,2,'bf',0,50)

    sb_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    sb_odds = [0.4403197520854301, 0.5879129571213885, 0.8460708229366423, 0.9963299568691596, 1.08006534945437,
               1.013246952983177, 0.9865720411215158, 1.012556244066168, np.nan, 1.070434864069297, 1.0988974789406314,
               1.1414296869397431,1.2072813297681462, 1.2460066546765498, 1.2039247393020196, 1.141113337756659,
               1.1280279965717268, 1.1141625616072954]
    sb = byhex(hex_filter,[],sb_range,sb_odds,34,45,'sb',0,90)

    arange = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    aodds = [0.1727002180482034, 0.1723182471491432, 0.2436916307320752, 0.3263790609957885,
             0.3919688855044393, 0.4968247633838836, 0.6937305113103779, 0.8223965979079845,
             1.1132992528771155, 1.1380693067242396,
             1.1164163204023465, 1.067729230927584, 1.007863563411918, 0.9205033160226128,
             0.9890150940355104, 1.063904738430273]

    age = byhex(hex_filter,[],arange,aodds,70,79,'age',0,150)

    drange = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
              80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
              150, 155, 160, 165, 170, 175, 180, 185, 190, 195]
    dodds = [np.nan, np.nan, 0.5011836165808218, 0.5215411169521906, 0.5819160003393484, 0.7248854990073924,
             0.6334098886987041, 0.4427610296480064, 0.6680794133476263, 0.4582263568571602, 0.1150653346223885,
             0.1147531203990016, 0.1553916652062251, 0.2485172497196862, 0.3208463216311091, 0.3832758832207134,
             0.3765696881894101, 0.242983711659716, 0.2038547229199977, 0.1484702224131105, 0.1208746066192809,
             0.0860634937238241, 0.086721434041713, 0.1035169486025075, 0.096932068656976, 0.0620879009095176,
             0.0482687052635198, 0.0574368819193491, 0.0518261075223342, 0.0454090574652511, 0.053011589884527,
             0.0508469447314887, 0.046253973540386, 0.0400042726809394,
             0.0540497184731681, 0.0502404247436967, 0.0308161917425344, 0.0193732158036937,
             0.0157075460563445, 0.0153369159425762]

    hex_filter['cp_t'] = hex_filter['cp']*0.001
    dist = byhex(hex_filter,[],drange,dodds,0,10,'cp_t',0,200)

    jrange = [-29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17]
    jodds = [3.7673728126068378, 1.7374397494145215, 0.6116233071486046, 0.4030079468821615,
             0.3531316995329558, 0.279584486750518, 0.2157590864001274, 0.1260759117326016,
             0.113508061240686, 0.1057474141066443, 0.1694503266403262, 0.4286629293841357,
             0.256763257760892]
    
    jan = byhex(hex_filter,[],jrange,jodds,-26.37,-25.26,'mj',-29,-17)

    normal_stand = get_comparison(df)
    print(normal_stand)

    ones = list(np.ones(len(normal_stand)))

    all_other = []
    nans = [] 

    for hex_orig in hex_list:
        if hex_orig not in normal_stand:
            all_other.append(hex_orig)
            nans.append(np.nan)

    append_hex = normal_stand + all_other
    normal_ones = ones + nans

    ndf = pd.DataFrame()
    ndf['hex'] = append_hex
    ndf['lookup_norm'] = normal_ones
    
    
    fig, ax = plt.subplots(1,2)

    gdf = gpd.read_file('NAMAP.shp') #.to_crs('EPSG:9001')
    all_other = gdf[gdf['postal'] != 'ON']
    water = gpd.read_file('LAKE.shp')
    water2 = gpd.read_file('ocean_box_corr.shp')

    #gdf['risk'] = elev + bf + age
    
    asm = gpd.read_file('grid_5000.shp') #.to_crs('EPSG:4326')

    hex_codes = list(hex_filter['hex'])

    joint_risk = np.prod(np.vstack([elev,bf,sb,age,jan]), axis=0).tolist()
    joint_risk2 = np.prod(np.vstack([elev,bf,sb,age,jan,dist]), axis=0).tolist()
    print(joint_risk2)

    hfill=[]
    rfill = [] 

    for hex_orig in hex_list:
        if hex_orig not in hex_codes:
            hfill.append(hex_orig)
            rfill.append(np.nan)


    df_fix = pd.DataFrame()
    df_fix['hex'] = hex_codes + hfill
    df_fix['risk'] = joint_risk + rfill
    df_fix['risk_sa'] = joint_risk2 + rfill
    df_fix['cp_t'] = list(hex_filter['cp_t']) + rfill
    #df_fix = df_fix.sort_values(by='id')

    
    #df_fix.cp_t[df_fix.hex ==336] = 10000000000
    print(df_fix)
    asm['hex'] = asm['id']
    #asm = asm.merge(df_fix, on='id', how='outer')
    asm =  pd.merge(asm, df_fix, how="left", on=["hex"])

    asm_norm =  pd.merge(asm, ndf, how="left", on=["hex"])
    print(asm)
    #asm['risk'] = df_fix['risk']
    #asm = asm.dropna(how='any')

    a = asm.plot('risk',ax=ax[0],cmap='RdGy_r',edgecolor='k',linewidth=0.25,
             zorder=14, alpha=0.5, legend=False)

    gdf.plot(ax=ax[0],facecolor="none",edgecolor='k',linewidth=0.5,
             zorder=15, alpha=1)
    water.plot(ax=ax[0],facecolor="k",edgecolor='k',linewidth=0.5,
             zorder=17, alpha=1)
    water2.plot(ax=ax[0],facecolor="k",edgecolor='k',linewidth=0.5,
             zorder=17, alpha=1)
    all_other.plot(ax=ax[0],facecolor="#FFFFFF",edgecolor='k',linewidth=0.5,
             zorder=16, alpha=1)
    a = asm_norm.plot('lookup_norm',ax=ax[0],facecolor="none",edgecolor='k',linewidth=1,
             zorder=15, alpha=0.5, legend=False)
    #ax[0].set_title('A')

    a = asm.plot('risk_sa',ax=ax[1],cmap='RdGy_r',edgecolor='k',linewidth=0.25,
             zorder=14, alpha=0.5, legend=False,vmin=asm.risk_sa.min(), vmax=asm.risk_sa.max())
    a = asm_norm.plot('lookup_norm',ax=ax[1],facecolor="none",edgecolor='k',linewidth=1,
             zorder=15, alpha=0.5, legend=False)

    gdf.plot(ax=ax[1],facecolor="none",edgecolor='k',linewidth=0.5,
             zorder=16, alpha=1)
    water.plot(ax=ax[1],facecolor="k",edgecolor='k',linewidth=0.5,
             zorder=17, alpha=1)
    water2.plot(ax=ax[1],facecolor="k",edgecolor='k',linewidth=0.5,
             zorder=17, alpha=1)
    all_other.plot(ax=ax[1],facecolor="#FFFFFF",edgecolor='k',linewidth=0.5,
             zorder=16, alpha=1)
    #ax[1].set_title('B')

    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    import matplotlib.colors as colors

    norm = colors.Normalize(vmin=asm.risk.min(), vmax=asm.risk.max())
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdGy_r')

    # add colorbar
    ax_cbar = fig.colorbar(cbar, ax=ax[0],fraction=0.046, pad=0.04)
    # add label for the colorbar
    #ax_cbar.set_label('Odds Ratio')

    norm = colors.Normalize(vmin=asm.risk_sa.min(), vmax=asm.risk_sa.max())
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdGy_r')

    # add colorbar
    ax_cbar = fig.colorbar(cbar, ax=ax[1],fraction=0.046, pad=0.04)
    # add label for the colorbar
    ax_cbar.set_label('Odds Ratio')

    for axes in ax[:2]:
        axes.set_aspect(1)


    xlim = ([asm.total_bounds[0]-40000,  asm.total_bounds[2]+40000])
    ylim = ([asm.total_bounds[1]-1,  asm.total_bounds[3]+1])

    ax[0].spines["top"].set_linewidth(1.25)
    ax[0].spines["left"].set_linewidth(1.25)
    ax[1].spines["top"].set_linewidth(1.25)
    ax[1].spines["left"].set_linewidth(1.25)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)    
    plt.show()

    fig.savefig('march8_plots/check_dpi2.svg', format='svg', dpi=1300)
    fig.savefig('march8_plots/check_dpi2.eps', format='eps', dpi=1300)

    
    

    
