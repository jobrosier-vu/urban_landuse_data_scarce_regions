# -*- coding: utf-8 -*-
"""
From labelled shapefiles create training data in tiles
Each tile is saved as .npy file in the folder corresponding to the land use label.
of the shapefile.

@author: JJR226
"""

import geopandas as gpd
import os
import numpy as np
import shapely
import rioxarray as rio 

# basepath
project_path = r''

# list of cities
cities = []

# define tile size
tile_size = 150

# define tile spacing
tile_spacing = 30

for city in cities:    
    # ground truth shapefiles
    base_path_osm = os.path.join(project_path,"DATA","SHAPEFILES")
    # destination path
    base_path_dest = os.path.join(project_path,'DEEPLEARNING','data','{}_Planet_150px_30sp_3res'.format(city))
    # satellite data
    base_path_satellite =  os.path.join(project_path,'DATA','SATELLITE',city,'{}_merged_2021_2023'.format(city),"composite.tif")
    
    # load raster
    loaded_raster = rio.open_rasterio(base_path_satellite)
    
    # define categories
    categories = ['commercial_institutional',
     'formal_residential',
     'industrial',
     'informal',
     'unbuilt']
        
    # per land use save the samples
    for category in categories:
        
        # load shapefiles
        dataset_lu = gpd.read_file(os.path.join(base_path_osm,city,'LU_CATEGORY',category,"shapes.shp"))
        dataset_lu = dataset_lu.to_crs(loaded_raster.rio.crs )
        
        # create folder if doesn't exist
        dest_folder = os.path.join(base_path_dest,category)
        os.makedirs(dest_folder, exist_ok=True)
        
        # loop over each polygon in the shapefile
        loc_identifyer = 0
        for loc in dataset_lu[dataset_lu.geometry != None].index:
            
            # buffer the shapefile
            xmin, ymin, xmax, ymax= dataset_lu.geometry[loc].buffer(200).bounds
            
            # clip the raster
            clipped_area = loaded_raster.rio.clip_box(minx=xmin,miny=ymin,maxx=xmax,maxy=ymax,crs=loaded_raster.rio.crs)
            
            xmin, ymin, xmax, ymax= dataset_lu.geometry[loc].bounds
            
            # create a grid of points with inter point space of 30m
            point_cells = []
            for x0 in np.arange(xmin, xmax+tile_spacing, tile_spacing ):
                for y0 in np.arange(ymin, ymax+tile_spacing, tile_spacing):
            
                    # bounds
                    point_cells.append( shapely.geometry.Point(x0, y0)  )  
            points = gpd.clip(gpd.GeoDataFrame({'loc_identifyer':[loc_identifyer]*len(point_cells),'geometry':point_cells}), dataset_lu.geometry[loc].buffer(-15))
            loc_identifyer+=1
            
            # sample the raster data for each point after buffering to get 150m cell size
            for sample in points.iloc:
                clipped = clipped_area.rio.clip([sample.geometry.buffer(tile_size/2).envelope])
                sample_values = clipped.values
                sample = gpd.GeoDataFrame({'loc_identifyer':[sample.loc_identifyer],'geometry':[sample.geometry]}).set_crs((loaded_raster.rio.crs)).to_crs(4326)
                sample_name = str(sample.geometry[0].y)+'_'+str(sample.geometry[0].x)+'_zone_'+str(sample.loc_identifyer[0])+'.npy'
                sample_loc = os.path.join(dest_folder,sample_name)
                
                np.save(sample_loc,sample_values)