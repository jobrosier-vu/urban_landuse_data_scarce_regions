# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:57:49 2024

@author: JJR226
"""
import pandas as pd

def divide_data(data, num_processes):
    # Divide the data into chunks based on the number of processes
    chunk_size = len(data) // num_processes
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    if len(data_chunks)>num_processes:
        temp_chunk = pd.concat([data_chunks[-2],data_chunks[-1]])
        data_chunks[-2] = temp_chunk
        data_chunks = data_chunks[:-1]
    return data_chunks

def devide_bf_data(BF_data,chunks):
    spatial_index = BF_data.sindex
    
    BF_chunks = []
    for chunk in chunks:
        possible_matches_index = list(spatial_index.intersection(chunk.geometry.buffer(500).total_bounds))
        possible_matches = BF_data.iloc[possible_matches_index]
        BF_chunks.append(possible_matches)
    return BF_chunks