# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:19:25 2024

@author: JJR226
"""
import os
from pathlib import Path
import json

def create_config_file(base_path_osm,base_path_config,city,Country):
    '''
    creates config file used in extracting the ROI from the osm file

    '''        
    # create the config file for osmium extract
    config_file = base_path_config+'/'+city+'/'+city+'.json'
    
    # create the output folder
    Path(os.path.join(base_path_osm,Country,city)).mkdir(parents=True, exist_ok=True)
    
    json_dict = {"directory":base_path_osm+'/'+Country+'/'+city+'/',"extracts":[{
          "output": city+".osm.pbf",
          "description": city,
          "polygon": {
            "file_name": "ROI.geojson",
            "file_type": "geojson"
          }}]
        }

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=2)
        
        
    return config_file