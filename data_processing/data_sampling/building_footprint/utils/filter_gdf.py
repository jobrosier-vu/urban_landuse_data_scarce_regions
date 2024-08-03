# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:57:07 2024

@author: JJR226
"""

import geopandas as gpd


def filter_gdf(gdf1,gdf2,mode):
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