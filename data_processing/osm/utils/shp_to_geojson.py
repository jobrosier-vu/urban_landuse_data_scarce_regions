# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:16:26 2024

@author: JJR226
"""
import geopandas as gpd
import os

# functions to extract osm file per city
def shp_to_gjson(base_path_config,city):
    """
    saves shapefile as geojson

    Parameters
    ----------
    base_path_config : file path
        base file path where the ROI.shp is found.
    city : string
        name of the city

    Returns
    -------
    None.

    """
    shp_file = gpd.read_file(os.path.join(base_path_config,city,'ROI.shp')).to_crs(4226)
    shp_file.to_file(os.path.join(base_path_config,city,'ROI.geojson'), driver='GeoJSON')