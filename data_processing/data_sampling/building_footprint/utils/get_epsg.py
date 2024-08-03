# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:13:06 2024

@author: JJR226
"""

def get_epsg(city):
    
    # epsg for europe
    epsg = 3035
    city=city.upper()
    if city == 'LUSAKA' or city == 'HARARE':
        epsg = 20935 # Lusaka
    elif city == 'KAMPALA' or city == 'NAIROBI' or city == 'DARESSALAAM':
        epsg = 21036 # Kampala/Nairobi
    elif city == 'LILONGWE':
        epsg = 20936  
    elif city == 'MAPUTO':
        epsg = 2737   
    elif city.upper() in ['SACREMENTO','HOUSTON','KANSAS','INDIANAPOLIS','NEWYORK','DENVER']:
        epsg = "ESRI:102039"
    return epsg

