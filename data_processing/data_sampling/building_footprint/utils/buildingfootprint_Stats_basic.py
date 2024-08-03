# -*- coding: utf-8 -*-
"""
building stats calculation based on: 
Angela Abascal, Ignacio Rodríguez-Carreño, Sabine Vanhuysse, Stefanos Georganos, Richard Sliuzas, Eleonore Wolff, Monika Kuffer,
Identifying degrees of deprivation from space using deep learning and morphological spatial analysis of deprived urban areas,
Computers, Environment and Urban Systems, Volume 95,2022,101820,ISSN 0198-9715, https://doi.org/10.1016/j.compenvurbsys.2022.101820.

@author: JJR226
"""

import numpy as np
import geopandas as gpd
import math
import os
from shapely.geometry import MultiPolygon, Polygon, LineString, box
from sklearn.neighbors import NearestNeighbors
import pandas as pd


# turn off default warning
pd.options.mode.chained_assignment = None  # default='warn'



def nr_vertices(poly):
    xx, yy = poly.exterior.coords.xy
    return len(xx)

# internal angle 
def vertices_from_polygon(poly):
    xx, yy = poly.exterior.coords.xy
    xx = list(xx)[:-1]
    yy = list(yy)[:-1]
    return list(zip(xx,yy))

def angle(x1, y1, x2, y2):
    # Use dotproduct to find angle between vectors
    # This always returns an angle between 0, pi
    numer = (x1 * x2 + y1 * y2)
    denom = math.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
    return math.acos(numer / denom) 


def cross_sign(x1, y1, x2, y2):
    # True if cross is positive
    # False if negative or zero
    return x1 * y2 > x2 * y1

def internal_irregular_angle(poly):
    """
    
    Returns (max(angles)-min(angles))/mean(angles)
    -------
    None.

    """
    poly = poly.simplify(tolerance=0.01)
    
    points = vertices_from_polygon(poly)
    
    angles = []
    for i in range(len(points)):
        p1 = points[i]
        ref = points[i - 1]
        p2 = points[i - 2]
        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]
        
        # Use dotproduct to find angle between vectors
        # This always returns an angle between 0, pi
        numer = (x1 * x2 + y1 * y2)
        denom = np.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        # make sure numer / denom is between -1,1
        ratio = numer / denom
        if ratio < -1 or ratio > 1:
            ratio = np.round(ratio)
        angle = math.acos(ratio) 
        angles.append(angle)

    irr_angle = (np.max(angles)-np.min(angles))/np.mean(angles)
    
    return irr_angle

def min_max_points_vertices(poly):
    #get points of minimum bounding rectangle
    mbr_points = list(zip(*poly.minimum_rotated_rectangle.exterior.coords.xy))
    
    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    major_axis = np.argmax(mbr_lengths)
       
    # get minor axis
    min_ax = np.min(mbr_lengths)    
    return major_axis,min_ax,mbr_points
    
def building_angle(poly):
    """
    Returns:
    Building angle with respect to the y-axis

    """     
    major_axis , _ , mbr_points = min_max_points_vertices(poly)
    
    #selected points
    mbr_points_selected = [mbr_points[major_axis],mbr_points[major_axis+1]]
    
    # set point with min y as p1
    p1_index = np.argmin(np.array(mbr_points_selected)[:,1])
    p2_index = np.argmax(np.array(mbr_points_selected)[:,1])
    
    #angle
    deltaX, deltaY = mbr_points_selected[p2_index][0] - mbr_points_selected[p1_index][0], mbr_points_selected[p2_index][1] - mbr_points_selected[p1_index][1]
    angle = (math.atan2(deltaY,deltaX)-math.pi/2)*180/math.pi
    
    return abs(angle)

def orientation_index(building_angles):
    """
    Returns:
    Building orientation index

    """      
    delta_angle = abs(np.array(building_angles[1:])-building_angles[0])
    OI = 1-abs(delta_angle-45)/45
    return OI

def nearest_neighbours(df):
    # get the central x,y from each building
    xx = df.geometry.centroid.x  
    yy = df.geometry.centroid.y
    # add them to the dataframe
    centroids = np.array(list(zip(xx,yy)))
    df['centroid'] = list(zip(xx,yy))
    
    # use nearest neighbours from sklearn to find the nearest 5 buildings or the buildings that are available
    n_neighbors = min(len(df),5)
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(centroids)
    return df, knn

def get_building_stats(BF_df):
    """
    
    Returns BF dataframe with building stats added per building
    -------
    Input BF dataframe

    """ 
    # check how many buildings are here and need at least 3 buildings
    if len(BF_df)<=2:
        setting='short'
        BF_df_centroids = BF_df
    else:
        setting = 'long'
        
        BF_df_centroids,knn = nearest_neighbours(BF_df)
    
       
    # initialize the building proximity, directionality, and inner irregularity vectors
    OI_mean = []
    OI_std = []

    in_irr = []
    amount_vertices = []
    shortest_vertices = []
    
    proximity_mean = []
    proximity_min = []
    
    proximity_mean = []
    proximity_min = []
    OI_mean = []
    OI_std = []
    
    # loop over the buildings to calculate directionality, proximity, and inner irregularity
    for building in BF_df_centroids.iloc:
        # get number of vertices
        ver = nr_vertices(building.geometry)
        amount_vertices.append(ver) 
        
        # get shortest vertices
        _ , short_vertices , _ = min_max_points_vertices(building.geometry)
        shortest_vertices.append(short_vertices)
        
        # inner irregularity
        in_irr.append(internal_irregular_angle(building.geometry))
        
        if setting=='long':
            # one get the 5 nearest buildings and their distance
            distance, building_index = knn.kneighbors([building.centroid], return_distance=True)
        
            # proximity
            proximity_mean.append(BF_df.iloc[building_index[0]].geometry.distance(BF_df.iloc[building_index[0][0]].geometry)[1:].mean())
            proximity_min.append(BF_df.iloc[building_index[0]].geometry.distance(BF_df.iloc[building_index[0][0]].geometry)[1:].min())
            
            
            # directionallity 
            building_angles = []
            for i in building_index[0]:
                building_angles.append(building_angle(BF_df.iloc[i].geometry))
            OI = orientation_index(building_angles)
        
            OI_std.append(np.std(OI))
            OI_mean.append(np.mean(OI))
            
        
        if setting=='short':
            proximity_mean = [0]*len(BF_df)
            proximity_min = [0]*len(BF_df)
            OI_mean = [0]*len(BF_df)
            OI_std = [0]*len(BF_df)
        
    # add the new data to the dataframe
    BF_df['OI_std'] = OI_std
    BF_df['OI_mean'] = OI_mean
    BF_df['proximity_mean'] = proximity_mean
    BF_df['proximity_min'] = proximity_min
    BF_df['in_irr'] = in_irr
    BF_df['nr_vertices'] = amount_vertices
    BF_df['shortest_vertices'] = shortest_vertices
    BF_df['area'] = BF_df.area
    
    return BF_df

def grid_agreggates(df):
    
    # num_buildings
    num_buildings = len(df)
    
    # area
    area_sum = df.area.sum()
    area_mean = df.area.mean()
    area_std = df.area.std()
    
    #vertices
    shortest_vertices_mean = df.shortest_vertices.mean()
    shortest_vertices_std = df.shortest_vertices.std()
    amount_vertices_mean = df.nr_vertices.mean()
    
    #inn_irr_angle
    in_irr_max = df.in_irr.max()
    in_irr_mean = df.in_irr.mean()
    
    # OI
    OI_mean_mean = df.OI_mean.mean()
    OI_std_mean = df.OI_std.mean()

    # dvl
    proximity_mean_mean = df.proximity_mean.mean()
    proximity_min_mean = df.proximity_min.mean()
    
    
    return [num_buildings,area_sum,area_mean,area_std,shortest_vertices_mean,shortest_vertices_std,amount_vertices_mean,in_irr_max,in_irr_mean,OI_mean_mean,OI_std_mean,proximity_mean_mean,proximity_min_mean]
