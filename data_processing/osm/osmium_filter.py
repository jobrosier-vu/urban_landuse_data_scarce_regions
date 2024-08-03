# -*- coding: utf-8 -*-
"""
with code from: 
    https://max-coding.medium.com/extracting-open-street-map-osm-street-data-from-data-files-using-pyosmium-afca6eaa5d00
    https://github.com/osmcode/pyosmium/blob/master/examples/amenity_list.py
    
@author: JJR226
"""

import osmium as osm
import pandas as pd
import os
import numpy as np
import pickle 
import geopandas as gpd
from pathlib import Path
import shapely.wkb as wkblib
import sys

base_path_osm = r"C:\Users\JJR226\Documents\PhD\Paper3\git_scripts\urbanlanduse_datascarceregions\pre_processing\osm"

sys.path.append(base_path_osm)

#preprocess
# regarding buildings the key: 'centre':institutional should be dropped
def preproc_areas(df):
    POIs_reclass_buildings = POIs_reclass
    POIs_reclass_buildings.pop('centre', None)
    
    
    split_names = [a.split(' ') if a is not None else a for a in list(df.name)]
    
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
            
            
class OSMHandler(osm.SimpleHandler):
    '''
    filter nodes, ways, and areas 

    '''       
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data = []
        self.streets = []
        self.areas = []
        self.LU_areas = []
          
    def node(self, n):
        if 'amenity' in n.tags or 'shop' in n.tags or 'tourism' in n.tags or 'landuse' in n.tags:
            lon, lat = n.location.lon, n.location.lat
            row = { "w_id": n.id, "amenity":n.tags.get("amenity"), "shop":n.tags.get("shop"), "tourism":n.tags.get("tourism"), "landuse":n.tags.get("landuse"), "lon":lon, "lat":lat}

            self.osm_data.append(row)


    def way(self, w):
        if 'highway' in w.tags:
            try:
                wkb = wkbfab.create_linestring(w)
                geo = wkblib.loads(wkb, hex=True)
            except:
                return
            row = { "w_id": w.id,'highway':w.tags.get("highway"), "geo": geo}

            self.streets.append(row)
    
    def area(self, a):
        if 'amenity' in a.tags or 'name' in a.tags or 'building':
            try:
                wkb = wkbfab.create_multipolygon(a)
                geo = wkblib.loads(wkb, hex=True)
            except:
                return            
            row = { "w_id": a.id, "amenity":a.tags.get("amenity"), "name":a.tags.get("name"),'building':a.tags.get('building'),"geo": geo }

            self.areas.append(row)
            
        if 'landuse' in a.tags:
            try:
                wkb = wkbfab.create_multipolygon(a)
                geo = wkblib.loads(wkb, hex=True)
            except:
                return            
            row = { "w_id": a.id, "landuse":a.tags.get("landuse"), "geo": geo }

            self.LU_areas.append(row)   

          
#reclass the dataframes
def reclass_buildings(df_osm_areas):
    df_osm_areas['amenity_LU'] = df_osm_areas['amenity'].map(POIs_reclass)
    df_osm_areas['building_LU'] = df_osm_areas['building'].map(POIs_reclass)
    df_osm_areas['name_LU'] = preproc_areas(df_osm_areas)
    # combine the amenity_LU and name_LU to form a single column
    df_osm_areas['amenity_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['name_LU']))] = df_osm_areas['name_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['name_LU']))]
    df_osm_areas['amenity_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['building_LU']))] = df_osm_areas['building_LU'][(pd.isnull(df_osm_areas['amenity_LU'])) & (~pd.isnull(df_osm_areas['building_LU']))]
    
    gdf_osm_areas = gpd.GeoDataFrame({'amenity_LU':df_osm_areas['amenity_LU'][~pd.isnull(df_osm_areas['amenity_LU'])],'w_id':df_osm_areas['w_id'][~pd.isnull(df_osm_areas['amenity_LU'])],'geometry':df_osm_areas['geo'][~pd.isnull(df_osm_areas['amenity_LU'])]})
    gdf_osm_areas = gdf_osm_areas.set_crs(4326)
    
    return gdf_osm_areas

def reclass_areas(df_osm_LU_areas):
    df_osm_LU_areas['landuse_LU'] = df_osm_LU_areas['landuse'].map(LU_reclass)
    gdf_osm_LU_areas = gpd.GeoDataFrame({'landuse_LU':df_osm_LU_areas['landuse_LU'],'landuse':df_osm_LU_areas['landuse'],'w_id':df_osm_LU_areas['w_id'],'geometry':df_osm_LU_areas['geo']})
    gdf_osm_LU_areas = gdf_osm_LU_areas.set_crs(4326)

    return gdf_osm_LU_areas

def reclass_POIS(df_osm_nodes):
    df_osm_nodes['amenity_LU'] = df_osm_nodes['amenity'].map(POIs_reclass)
    # wherever shop or tourism is defined fill in commercial as amenity_LU
    df_osm_nodes['amenity_LU'][(~pd.isna(df_osm_nodes['shop'])) | (~pd.isna(df_osm_nodes['tourism']))] = 'commercial'
    
    df_osm_nodes = df_osm_nodes[~pd.isna(df_osm_nodes['amenity_LU'])]
    
    gdf_osm_nodes = gpd.GeoDataFrame(df_osm_nodes,geometry=gpd.points_from_xy(df_osm_nodes.lon,df_osm_nodes.lat))
    gdf_osm_nodes = gdf_osm_nodes.set_crs(4326)

    return gdf_osm_nodes

if __name__ == "__main__": 

    # load poi dict
    dict_location = r"C:\Users\JJR226\Documents\PhD\Paper3\Data\OSM\General"
    with open(os.path.join(dict_location,"POIs_reclass_dictionary.pkl"), 'rb') as f:
        POIs_reclass = pickle.load(f)
    
    # load LU dict
    with open(os.path.join(dict_location,"LU_reclass_dictionary.pkl"), 'rb') as f:
        LU_reclass = pickle.load(f)

    
    # A global factory that creates WKB from a osmium geometry
    wkbfab = osm.geom.WKBFactory()
    
    for Country in os.listdir(base_path_osm):
        for city in next(os.walk(os.path.join(base_path_osm, Country)))[1]:
            
                
            # Initiate the osm handler class
            osmhandler = OSMHandler()
            
            # scan the input file and fills the handler list accordingly
            osmhandler.apply_file(os.path.join(base_path_osm,Country,city,city+'.osm.pbf'), locations=True)            
            
            #reclassify the osm data
            gdf_osm_nodes = reclass_POIS(pd.DataFrame(osmhandler.osm_data))
            gdf_osm_areas = reclass_buildings(pd.DataFrame(osmhandler.areas))
            gdf_osm_LU_areas = reclass_areas(pd.DataFrame(osmhandler.LU_areas))
            
            # create destination path if not exist
            Path(os.path.join(base_path_osm,Country,city,"Classified_check")).mkdir(parents=True, exist_ok=True)
            
            # save the dataframes per osm type to disk 
            gdf_osm_areas.to_file(os.path.join(base_path_osm,Country,city,"Classified_check","df_osm_areas_classified_quality.shp"))
            gdf_osm_LU_areas.to_file(os.path.join(base_path_osm,Country,city,"Classified_check","df_osm_LU_areas_classified_quality.shp"))
            gdf_osm_nodes.to_file(os.path.join(base_path_osm,Country,city,"Classified_check","df_osm_nodes_classified_quality.shp"))