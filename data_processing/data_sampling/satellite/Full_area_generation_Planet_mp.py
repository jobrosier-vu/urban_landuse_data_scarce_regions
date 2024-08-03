# -*- coding: utf-8 -*-
"""

loads a PlanetScope raster file. 
For each 150x150 tile the data is saved to a .npy file

@author: JJR226
"""


import geopandas as gpd
import os
import numpy as np
import rioxarray as rio 
import time
import multiprocessing as mp

def func(city_list):
    
    # basepath
    project_path = r''
    
    for city in city_list:
        ########### VHR
        base_path_dest = os.path.join(project_path,'')
        SEGMENTS_PATH = os.path.join(project_path,'')
        ROI_PATH = os.path.join(project_path,'')
        base_path_satellite = os.path.join(project_path,'')

        # create folder if doesn't exist
        os.makedirs(base_path_dest, exist_ok=True)
            
    
        loaded_raster = rio.open_rasterio(base_path_satellite)
        
        # import the segments    
        segments = gpd.read_file(SEGMENTS_PATH)
        segments = segments.to_crs(loaded_raster.rio.crs)
        
        # import ROI
        ROI = gpd.read_file(ROI_PATH)
        ROI = ROI.to_crs(loaded_raster.rio.crs)    

    
        # intersect the ROI and the Raster
        segments_select = gpd.sjoin(segments, ROI, how='inner', op='intersects')    
        for idx in range(len(segments_select)):
            minx,miny,maxx,maxy = segments_select.iloc[idx].geometry.buffer(2).bounds
            
            try:
                xdsc = loaded_raster.rio.clip_box(
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,)
                sample = xdsc.values[:,:50,:50]
                if (sample!=0).all() and sample.shape==(4, 50, 50):
                    name = 'sample_{}'.format(segments_select.iloc[idx].id)
                    sample_loc = os.path.join(base_path_dest,name)
                    np.save(sample_loc,sample)
            except:
                continue


def main(cities_list):     
    pool = mp.Pool(7)
    total_successes = pool.map(func, cities_list) # Returns a list of lists
    # Flatten the list of lists
    #total_successes = [ent for sublist in total_successes for ent in sublist]
    return 'finished'
        
if __name__ == '__main__':
    city_list = [['DARESSALAAM']]
    
    start = time.time()
    main(city_list)
    end = time.time()
    print(end - start)