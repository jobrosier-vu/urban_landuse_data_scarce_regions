# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:48:54 2023

@author: JJR226
"""
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt

def plot_cm(y_true, y_pred,save_name, Title,categories, figsize=(8,8),dpi=100):
    
    oa =  np.sum(np.array(y_true)==np.array(y_pred))/len(y_pred)*100
    target_names = categories
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    avg_f1 = sum(2*(recall*precision)/(recall+precision))/len(recall)*100
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.subplots(figsize=figsize)
    plt.title(Title+', OA: {:.2f}% F1: {:.2f}%'.format(oa,avg_f1))
    sns.heatmap(cm, cmap= "Blues", annot=annot, fmt='', vmin=0, vmax=100,cbar_kws={'format': '%.0f%%', 'ticks': [0,20,40,60,80,100]},)  
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        tick_marksy = np.arange(0.5,len(target_names),1)
        plt.xticks(tick_marksy, target_names, rotation=45,ha='right')
        plt.yticks(tick_marksy, target_names, rotation=0)  
    plt.tight_layout()
    plt.savefig(os.path.join(save_name,Title))