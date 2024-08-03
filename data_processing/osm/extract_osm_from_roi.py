# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:19:38 2022

extract the OSM data in a region of interest from .osm.pbf file

"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys

base_path = r""

base_path_osm = os.path.join(base_path,'pre_processing','osm')
base_path_data = os.path.join(base_path,'data','osm')
base_path_config = os.path.join(base_path,'data','shapefiles')

sys.path.append(base_path_osm)

from base_path_osm.utils import create_config_file, osmium_extract_ROI, shp_to_gjson

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--country", help="Country name first letter capitalized")
parser.add_argument("-p", "--place", help="place name first letter capitalized")
args = vars(parser.parse_args())

Country = args["country"]
city = args["place"] 

if __name__ == "__main__":
    shp_to_gjson(base_path_config,city)
    config_file = create_config_file(base_path_data,base_path_config,city,Country)
    osmium_extract_ROI(config_file,base_path_osm,Country)