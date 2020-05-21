# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:52:25 2020

@author: Thomas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import scipy.ndimage as nd
from skimage.measure import regionprops

from helferlein import imcolor, files

# cont = np.load(r'D:\data science\cont.npy')

# def fahrenheit2celsius(arr):
#     return (arr-32) * 5/9


# definition der funktion fileselect; als input wert gibst du ihm nur das jahr an welches du reinladen möchtest
# du musst den Pfad anpassen, also anstatt D:\data science\Archive... den pfad angeben, wo die matlab files liegen
def fileselect(a = 2007):
    
    path_nasa = r'D:\data science\Archive_{}'.format(str(a))
    data_names = files(path_nasa)
    x = np.array([loadmat(path_nasa+os.sep+i)['NASAwf'][0] for i in data_names])
    
    labels = list(x[0].dtype.names)
    xdata = np.array([i[0] for i in x])
    
    data_dic = {}
    for i in labels: data_dic[i] = []
    
    for i in range(len(xdata)):
        for j in range(len(xdata[i])):
            data_dic[labels[j]].append(xdata[i][j])
    return data_dic

# hier wird die funktion ausgeführt und verschiedene dataframes geladen
data_dic07 = fileselect(2007)
data_dic08 = fileselect(2008)
data_dic13 = fileselect(2013)

# tempmax = np.array(data_dic['TempMax'])
# tempmin = np.array(data_dic['TempMin'])

# plot(np.mean(tempmax, axis = (1,2)), 'temp change')
# plot(np.mean(tempmin, axis = (1,2)), 'temp change', clf = False)
# plt.legend(['tempmax', 'tempmin'])

# =============================================================================
# labs = nd.label(tempmax[0])[0]
# cont = np.zeros_like(labs)
# for j,i in enumerate(peaks): cont[labs == i] = j+1
# imcolor(cont, 'cont')
# =============================================================================

# propstmax = np.array([regionprops(cont, i) for i in tempmax])

# imgtmax = np.array([[i.intensity_image for i in j] for j in propstmax])
# regtmax = imgtmax.T

# meantmax = np.array([[i.mean_intensity for i in j] for j in propstmax])

# =============================================================================
# def extract_mean_feature(data_dic, feature = 'TempMax'):
#     feat = np.array(data_dic[feature])
#     
#     props = np.array([regionprops(cont, i) for i in feat])
#     # img = np.array([[i.intensity_image for i in j] for j in props])
#     mean = np.array([[i.mean_intensity for i in j] for j in props])
#     
#     return mean
# 
# plot(fahrenheit2celsius(extract_mean_feature(data_dic13, 'TempMax')))
# 
# =============================================================================


# fig, ax = plt.subplots(4,5)
# a = ax.flatten()

# for i in range(20): 
    
#     a[i].imshow(nasa[0][i], cmap = 'nipy_spectral')
#     a[i].set_title(labels[i])


