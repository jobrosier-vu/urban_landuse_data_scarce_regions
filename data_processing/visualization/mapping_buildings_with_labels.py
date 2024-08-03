# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:53:29 2023

@author: JJR226
"""

import geopandas as gpd
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from joblib import Parallel, delayed

def func(inputs):
    '''

    For each segment find the buildings that are intersecting. Buildings can therefore belong to multiple segments. To find the segment that covers the building
    to the greatest extend we clip eacht building and save the area clipped. By noting the building index together with the area we can later select the segment which has the largest coverage.

    '''
    stats_list = [] 
    geometry_list = []
    index_list = []
    area_list =[]
    for i in inputs:
        
        possible_matches_index = list(spatial_index.intersection(labelled_grid.iloc[i].geometry.bounds))
        possible_matches = BF.iloc[possible_matches_index].to_crs(3857)
        
        clipped = possible_matches.clip(labelled_grid.iloc[i].geometry).to_crs(3857)
        
        if len(possible_matches)>0:
            # labels
            stats_list.extend([labelled_grid.iloc[i].label_deta]*len(possible_matches))
            # original geometry
            geometry_list.extend(clipped.geometry.to_list())
            # index
            index_list.extend(possible_matches.FID.to_list())
            # clipped area 
            area_list.extend(clipped.area)

        else:
            
            stats_list.extend([None])
            geometry_list.extend([None])
            index_list.extend([None])
            area_list.extend([None])
            
    total = [stats_list,geometry_list,index_list,area_list]    
    return total

def main():     
    pool = mp.Pool(6)
    total_successes = pool.map(func, [[*range(len(labelled_grid))]]) # Returns a list of lists
    # Flatten the list of lists
    #total_successes = [ent for sublist in total_successes for ent in sublist]
    return total_successes

if __name__ == '__main__':
    

    BF = gpd.read_file(r"")
    
    labelled_grid = gpd.read_file(r"")
    
    #import the segments    
    labelled_grid = labelled_grid.to_crs(BF.crs)
    
     #get spatial indexing of BF data
    spatial_index = BF.sindex
    
    # apply the func
    total_successes = func([*range(len(labelled_grid))])
    
    bf_labelled = gpd.GeoDataFrame({'indexes':total_successes[2],'label_deta':total_successes[0],'area':total_successes[3],'geometry':total_successes[1]})
    bf_labelled = bf_labelled.dropna(axis='rows')   
    bf_labelled = bf_labelled.set_crs(3857)
     
    # keep the largest area
    bf_labelled = bf_labelled.sort_values(by=['indexes','area'])
    bf_labelled = bf_labelled.drop_duplicates(subset='indexes', keep='last')
    
