# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:04:47 2023

@author: JJR226
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def split_category(activations_path,category):
    """
    Splits a catogry in two sub-categories based on the activations of the
    deep learning approach or statistics from building footprints. 
    Clustering is done using kmeans approach. 

    Parameters
    ----------
    activations_path : path
        path where grid like shapefile is located.
    category : integer
        integer that represents a land use category


    Returns
    -------
    pandas dataframe with labels split based on clustering

    """    
    # import features
    activations_labels = np.load(activations_path,allow_pickle=True)
    df = pd.DataFrame({'labels':activations_labels[0,:]*10,'locations':activations_labels[1,:],'activations':activations_labels[2,:]})
    df['detailed_labels'] = df.labels
    
    #select one class
    class_collection = df[df.detailed_labels==category]
    
    # standardize features
    features = np.array(class_collection.activations)
    features = np.vstack( features )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    
    # apply the k-means 10 times
    kmeans_kwargs = {
         "init": "random",
         "n_init": 10,
         "max_iter": 1000,
         "random_state": 11,
     }
    
    sse = []
    for k in range(1, 5):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        
    #
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 5), sse)
    plt.xticks(range(1, 5))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    
    
    kmeans = KMeans(2)
    new_labels = kmeans.predict(scaled_features)+category
    
    #df[df['detailed_labels']==category].detailed_labels=new_labels
    
    for i,value in zip(df[df['detailed_labels']==category].index,new_labels):
        df.loc[i,'detailed_labels'] = value
    
    df.detailed_labels = df.detailed_labels.replace(10,100)
    return df


if __name__ == "__main__":
    activations_path = r''
    save_path = r''
    
    category = 1 # residential
    df = split_category(activations_path,category)
    np.save(save_path,df.to_numpy())


