# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:22:48 2024

@author: JJR226
"""
import os
import subprocess

# run osmium extract

def osmium_extract_ROI(config_file,base_path_osm,Country):
    '''
    uses config file to extract ROI from osm file

    '''    
    input_file = os.path.join(base_path_osm,Country,Country.lower()+"-latest.osm.pbf")
    list_files = subprocess.run(["osmium", "extract","-c",config_file,input_file])
    print("The exit code was: %d" % list_files.returncode)