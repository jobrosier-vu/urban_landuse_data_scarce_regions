# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:28:29 2024

@author: JJR226
"""

import geopandas as gpd
import numpy as np
import pandas as pd


def preproc_areas(df,POIs_reclass_buildings):
    
    # centre needs to be removed from list 
    POIs_reclass_buildings.pop('centre', None)
    
    split_names = [a.split(' ') if a is not None else a for a in list(df.name)]
    
    # for each label in the name, try to reclassify them, if there are multiple select only the first one
    names_reclassified = []
    for label_list in split_names:
        if label_list is None:
            names_reclassified.append(None)
        else: 
            reclass = []
            for label in label_list:
                try:
                    test = POIs_reclass_buildings[label.lower()]
                    reclass.append(test)    
                except:
                    continue
            if len(np.unique(reclass))==1:
                names_reclassified.append(reclass[0])
            else:
                names_reclassified.append(None)
    return names_reclassified

#reclass the dataframes
def reclass_buildings(df_osm_areas,reclass_dict):
    
    # reclass amenities
    df_osm_areas['amenity_LU'] = df_osm_areas['amenity'].map(reclass_dict)
    # reclass buildings
    df_osm_areas['building_LU'] = df_osm_areas['building'].map(reclass_dict)
    # reclass building names
    df_osm_areas['name_LU'] = preproc_areas(df_osm_areas)
    # combine the amenity_LU and name_LU to form a single column
    df_osm_areas['amenity_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['name_LU']))] = df_osm_areas['name_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['name_LU']))]
    df_osm_areas['amenity_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['building_LU']))] = df_osm_areas['building_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['building_LU']))]
    
    gdf_osm_areas = gpd.GeoDataFrame({'amenity_LU':df_osm_areas['amenity_LU'][~pd.isnull(df_osm_areas['amenity_LU'])],'w_id':df_osm_areas['w_id'][~pd.isnull(df_osm_areas['amenity_LU'])],'geometry':df_osm_areas['geo'][~pd.isnull(df_osm_areas['amenity_LU'])]})
    gdf_osm_areas = gdf_osm_areas.set_crs(4326)

    
    return gdf_osm_areas