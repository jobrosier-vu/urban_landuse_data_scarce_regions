import multiprocessing
import geopandas as gpd
import pandas as pd
import numpy as np
import sys
import os

BASEPATH = r'C:\Users\JJR226\Documents\PhD\paper4\SCRIPTS\PREPROCESSING\BUILDINGFOOTPRINTS'

sys.path.append(BASEPATH)

from UTILS.buildingfootprint_Stats_basic import *

def filter_gdf(gdf1,gdf2,mode='excluding'):
    '''
    

    Parameters
    ----------
    gdf1 : geodataframe
        gdf to be filtered.
    gdf2 : geodataframe
        the filter.

    Returns
    -------
    gdf1 : geodataframe
        filtered dataframe.

    ''' 
    
    gdf1['savedindex']= gdf1.index
    try:
        gdf2 = gdf2.drop('savedindex', axis=1)
    except:
        pass
    
    intersecting = gdf2.sjoin(gdf1, how='inner')['savedindex']
    if mode == 'including':
        gdf1_selected = gdf1[gdf1.savedindex.isin(intersecting)] 
    if mode == 'excluding':
        gdf1_selected = gdf1[~gdf1.savedindex.isin(intersecting)] 
    gdf1_selected = gdf1_selected.drop('savedindex', axis=1)
    
    return gdf1_selected

def process_tile(tile, building_data):
    buffer_size = 0
    
    # Perform processing on the given tile and building data
    
    spatial_index = building_data.sindex
    
    results = []
    for i in range(len(tile)):
        segment = tile.iloc[i]
        possible_matches_index = list(spatial_index.intersection(segment.geometry.buffer(buffer_size).bounds))
        possible_matches = building_data.iloc[possible_matches_index]
        
        try:
            BF_stats_area = get_building_stats(possible_matches)
            if len(BF_stats_area)>0:
                stats = grid_agreggates(BF_stats_area)
                # remove any nans and replace by zero
                stats = [0 if np.isnan(x) else x for x in stats]
            else:
                stats = list(np.zeros(13))  
                
        except:
            stats = np.full((1,13),np.nan)[0].tolist()
            print('Error in tile:',segment.id)            
            continue
        

        results.append([stats,segment.id])
        
    return results

def divide_data(data, num_processes):
    # Divide the data into chunks based on the number of processes
    chunk_size = len(data) // num_processes
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    if len(data_chunks)>num_processes:
        temp_chunk = pd.concat([data_chunks[-2],data_chunks[-1]])
        data_chunks[-2] = temp_chunk
        data_chunks = data_chunks[:-1]
    return data_chunks

def devide_bf_data(BF_data,chunks):
    spatial_index = BF_data.sindex
    
    BF_chunks = []
    for chunk in chunks:
        possible_matches_index = list(spatial_index.intersection(chunk.geometry.buffer(150).total_bounds))
        possible_matches = BF_data.iloc[possible_matches_index]
        BF_chunks.append(possible_matches)
    return BF_chunks
        
    
def load_building_data(city):
    
    setting = 'STATS'
    
    if setting=='DL':
        # data paths
        BF_PATH = r'C:\Users\JJR226\Documents\PhD\paper4\DATA\BUILDINGFOOTPRINT\{}\building_footprints_{}_full.shp'.format(city.upper(),city)
        ROI_PATH = r'C:\Users\JJR226\Documents\PhD\paper4\DATA\SHAPEFILES\{}\ROI\ROI.shp'.format(city.upper())
        
        # load the BF and temp save the crs
        BF = gpd.read_file(BF_PATH)
        BF = BF[BF.geom_type != 'MultiPolygon']
        BF_crs = BF.crs
        
        # get only the geometry
        BF_list = list(BF.geometry)
        BF = gpd.GeoDataFrame(geometry=BF_list).set_crs(BF_crs)
        
        # import ROI
        ROI = gpd.read_file(ROI_PATH)
        ROI = ROI.to_crs(BF.crs)   
        
        # filter BF
        BF = filter_gdf(BF,ROI,mode='including')
        
        # get correct epsg
        epsg = get_epsg(city)
        
        BF = BF.to_crs(epsg=epsg)
        
    if setting=='STATS':
        # data paths
        BF_PATH = r'C:\Users\JJR226\Documents\PhD\paper4\DATA\BUILDINGFOOTPRINT\{}\building_footprints_{}_full.shp'.format(city.upper(),city)
        
        # load the BF and temp save the crs
        BF = gpd.read_file(BF_PATH)
        BF = BF[BF.geom_type != 'MultiPolygon']
        BF_crs = BF.crs
        
        # get only the geometry
        BF_list = list(BF.geometry)
        BF = gpd.GeoDataFrame(geometry=BF_list).set_crs(BF_crs)
        
        # get correct epsg
        epsg = get_epsg(city)
        
        BF = BF.to_crs(epsg=epsg)
        
    return BF

def generate_grid(base_path,city):

    SEGMENTS_PATH = os.path.join(base_path,'SHAPEFILES',city.upper(),'ROI','Grid_classification.shp')
    # get correct epsg
    epsg = get_epsg(city)    

    # import the segments
    segments = gpd.read_file(SEGMENTS_PATH)
    
    if 'id' not in segments.columns:
     ids = list(segments.index)
     segments['id'] = ids

    segments = segments[['id','geometry']]
    segments = segments.to_crs(epsg=epsg)   
    
    return segments
    
def get_epsg(city):
    
    # epsg for europe
    epsg = 3035
    
    if city == 'Lusaka' or city == 'Harare':
        epsg = 20935 # Lusaka
    elif city == 'Kampala' or city == 'Nairobi' or city == 'Daressalaam':
        epsg = 21036 # Kampala/Nairobi
    elif city == 'Lilongwe':
        epsg = 20936  
    elif city == 'Maputo':
        epsg = 2737   
    return epsg
    
if __name__ == "__main__":
    # List of cities you want to process
    cities = ['Newyork','Houston'] 
    
    base_path = r'C:\Users\JJR226\Documents\PhD\paper4\DATA'
    
    # building footprint data for each city 
    building_footprint_data = {city: load_building_data(city) for city in cities}

    # Dummy grid data (replace with your actual grid data)
    grid_data = {city: generate_grid(base_path,city) for city in cities}

    # Number of processes to use
    num_processes = 11

    
    for city in cities:
        processes = []
        
        # location to write the data to
        SAVE_PATH = r"C:\Users\JJR226\Documents\PhD\paper4\DATA\BUILDINGFOOTPRINT\{}\STATISTICS\FULLAREA".format(city.upper())
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        # Divide the grid into chunks
        grid_chunks = divide_data(grid_data[city], num_processes)

        # Divide the building footprint data into chunks
        building_data_chunks = devide_bf_data(building_footprint_data[city], grid_chunks)


        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_tile, [(section[0], section[1]) for _, section in enumerate(zip(grid_chunks, building_data_chunks))])

        stats = [x[0] for xs in results for x in xs]
        names = [x[1] for xs in results for x in xs]
        
        total_np = pd.DataFrame({'names':names,'stats':stats})
        
        np.save(os.path.join(SAVE_PATH,'MS_buildingfootprint_stats.npy'),total_np)
