# -*- coding: utf-8 -*-
"""
Calculating statistics from building footprints. 
For each city the samples are loaded that were created using the script "sampling_ground_truth_satellite.py"
In order to speed up the process we match each sample with its original ground truth polygon
Per polygon the building footprint data is selected and per tile the building footprint stats are calculated
All stats are then saved to a pickle file. 

 
"""
import numpy as np
import geopandas as gpd
import os
import shapely
import pandas as pd
import multiprocessing as mp

import sys, os

BASEPATH = r''

sys.path.append(BASEPATH)

from utils.buildingfootprint_Stats_basic import *

# turn off default warning
pd.options.mode.chained_assignment = None  # default='warn'

base_path = r''


categories = ['commercial_institutional',
 'formal_residential',
 'industrial',
 'informal',
 'unbuilt']

def get_zones(samplelist):
    '''
    Parameters
    ----------
    samplelist : list of samples from Planet dataset.

    Returns
    -------
    zonelist : list of zones extracted from name.
    point_list : list of shapely points extracted from name.

    '''
    zonelist = []
    point_list = []
    for sample in samplelist:
        zone = sample.split('_')[-1][:-4]
        zonelist.append(int(zone))
        
        lat, lon = float(sample.split('_')[0]),float(sample.split('_')[1])
        point_list.append(shapely.geometry.Point(lon, lat))
        
    return zonelist,point_list
   
def get_stats_from_city(cities):
    
    # define the tile size nxn
    sample_size = 300
    
    for city in cities:
        
        # get the epsg code per city
        epsg = get_epsg(city)  
            
        base_path_samples = os.path.join(base_path,'DEEPLEARNING','DATA','{}_Planet_150px_30sp_3res'.format(city.upper()))
        
        # load the building footprints 
        BF = gpd.read_file(os.path.join(base_path,'DATA','BUILDINGFOOTPRINT',city.upper(),'building_footprints_{}.shp'.format(city)))
        BF = BF.to_crs(epsg)  
        BF = BF[BF.geom_type != 'MultiPolygon']
        
        # get spatial indexing of BF data
        spatial_index = BF.sindex
    
        for category in categories:   
            # load the zones per category
            zones_lu = gpd.read_file(os.path.join(base_path,'DATA','SHAPEFILES',city.upper(),'LU_CATEGORY',category,"shapes.shp"))
            zones_lu = zones_lu.to_crs(epsg)
            # remove any zones without geometry
            zones_lu = zones_lu[zones_lu.geometry != None].reset_index(drop=True)
            
            # create folder if doesn't exist
            os.makedirs(os.path.join(base_path,'DATA','BUILDINGFOOTPRINT',city.upper(),'STATISTICS','{}m'.format(sample_size)), exist_ok=True)
            
            # from the saved training/testing/validation folder get the samples belonging to the category
            samplelist = os.listdir(os.path.join(base_path_samples,category))
            # for each sample identify the zone from which it came
            zonelist,point_list = get_zones(samplelist)
            # create a dataframe with samples, zones and the centroid 
            zone_sample_df = gpd.GeoDataFrame({'samples':samplelist,'zones':zonelist,'geometry':point_list})
            # translate to crs of the building footprint data
            zone_sample_df = zone_sample_df.set_crs(4326).to_crs(epsg)
            
            category_list = []
            name_list = []
            zone_list = []
            stats_list = []
        
            for zone in np.unique(zonelist):
                # get all samples that belong to the selected zone
                selected_df = zone_sample_df[zone_sample_df.zones==zone]
                
                
                # get the building stats per sample
                for selected_sample in selected_df.iloc:
                    
                    # get the bounding box of the buffered centroid
                    possible_matches_index = list(spatial_index.intersection(selected_sample.geometry.buffer(sample_size/2).bounds))
                    possible_matches = BF.iloc[possible_matches_index]
                    
                    # save name
                    name = selected_sample.samples
                    if len(possible_matches)>0:
                        BF_stats_sample = get_building_stats(possible_matches)
                        stats = grid_agreggates(BF_stats_sample)
                        # replace any nans with zeros
                        stats = [0 if np.isnan(x) else x for x in stats]
                    else:
                        stats = list(np.zeros(13))
                    # fill the lists with sample name, category, zone, stats
                    category_list.append(category)
                    name_list.append(name)
                    zone_list.append(zone)
                    stats_list.append(stats)
                print('{} out of {} finished'.format(zone,len(np.unique(zonelist))))
                
                
            bf_stats = pd.DataFrame({'names':name_list,'zones':zone_list,'stats':stats_list})
            bf_stats.to_pickle(os.path.join(base_path,'DATA','BUILDINGFOOTPRINT',city.upper(),'STATISTICS','{}m'.format(sample_size),"intersect_{}.pkl".format(category)))    
    return 'finished'

def main(cities_list):     
    pool = mp.Pool(4)
    total_successes = pool.map(get_stats_from_city, cities_list) # Returns a list of lists

    return 'finished'
        
if __name__ == "__main__":    
    cities = []
    sample_size = 300
    
    main(cities)