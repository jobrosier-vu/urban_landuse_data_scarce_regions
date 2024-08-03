# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:55:08 2023

@author: JJR226
"""
import geopandas as gpd
import numpy as np


def pred_to_grid(SEGMENTS_PATH,PREDICTION_PATH,SAVE_PATH):
    """
    

    Parameters
    ----------
    SEGMENTS_PATH : path
        path where grid like shapefile is located.
    PREDICTION_PATH : path
        .npy or .plk file where the prededictions from the deep learning model
        is located.
    SAVE_PATH : path
        .shp file location to store results.

    Returns
    -------
    None.

    """
    segments = gpd.read_file(SEGMENTS_PATH)
    segments.sort_values('id', inplace=True)
    
    # load predictions 
    predictions = np.load(PREDICTION_PATH,allow_pickle=True)
    pred_id = []
    pred_id_detailed = []
    
    # fill grid with classification results
    for seg in segments.id:
        try:
            idx = np.where(predictions[:,1] == int(seg))[0][0]
            pred_id.append(predictions[idx,0])
            pred_id_detailed.append(predictions[idx,3])
        except:
            pred_id.append(None)
            pred_id_detailed.append(None)
            
    segments['label'] = pred_id
    segments['label_detailed'] = pred_id_detailed # in case residential class was split
    segments.to_file(SAVE_PATH)
    
if __name__ == "__main__":
    # load segments
    SEGMENTS_PATH = r""
    PREDICTION_PATH = r""
    SAVE_PATH = r""
    
    pred_to_grid(SEGMENTS_PATH,PREDICTION_PATH,SAVE_PATH)