# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:28:43 2023

@author: JJR226
"""
import numpy as np
 
def combine_categories(datasets,combinations):
    
    # create a dict to link category names to integer values
    categories_original = dict(zip(datasets[0].dataset.dataset.classes,range(len(datasets[0].dataset.dataset.classes))))
    
    if len(combinations)==0:
        new_categories=categories_original
    
    else:
        samples = np.array(datasets[0].dataset.dataset.samples)
        targets = samples[:,1]
        targets = [int(x) for x in targets]
        # update the targets
        for combination in combinations:
            cat_nr = [categories_original[cat] for cat in combination]
            cat_nr = np.sort(np.array(cat_nr),0)        
            for nr in cat_nr[1:]: 
                targets = np.where(targets==nr,cat_nr[0],targets)
        
        # create new category strings        
        new_categories = [list(categories_original.keys())[i] for i in np.unique(targets)]
       
        # get targets in ascending with range
        for cat,new_cat in zip(np.unique(targets),list(range(len(np.unique(targets))))):
            if cat==new_cat:
                continue
            if cat!=new_cat:
                targets = np.where(targets==cat,new_cat,targets)
        
        #targets = [str(x) for x in targets]
        samples[:,1] = targets
        samples = [(str(d), int(s)) for d, s in list(samples)]
        
        
        # apply to all datasets  
        new_categories_dict = dict(zip(new_categories,list(range(len(np.unique(targets))))))
        for dataset in datasets:
            dataset.dataset.dataset.samples = samples
            dataset.dataset.dataset.classes = new_categories
            dataset.dataset.dataset.class_to_idx = new_categories_dict

    return datasets, new_categories