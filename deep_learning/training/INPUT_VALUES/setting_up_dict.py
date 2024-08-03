# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:59:25 2023

@author: JJR226
"""
import numpy as np
import os 

# input dict 
input_dict = {'model_name':'DenseNet42',
         'dataset_name':['NAIROBI_Planet_150px_30sp_3res','KAMPALA_Planet_150px_30sp_3res','LUSAKA_Planet_150px_30sp_3res'],
         'building_footprint':'total_df_300m.npy',
         'experiment_name':'Planet_BF300_lr2_SGD_weights_3_validation_12_alpha_05_three_cities',
         'combinations':[],
         'alpha':0.5,
         'val_test_size':[0.1,0.2],
         'batch_size':64,
         'num_workers':4,
         'optimizer':'SGD',
         'learning_rate':1e-2,
         'weight_increase':3,
         'seed':[4,5,6],
         'pretrained_network':None,
         'sample_size':[]
         }


savename = input_dict['model_name']+'_'+'three_cities'+'_'+input_dict['experiment_name']+'.npy'
savedir = r'C:\Users\JJR226\Documents\PhD\paper4\DEEPLEARNING\TRAINING\INPUT_VALUES'
np.save(os.path.join(savedir,savename), input_dict) 


#['mixed_residential_commercial','formal_residential']