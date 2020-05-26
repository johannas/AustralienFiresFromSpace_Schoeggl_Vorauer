# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:42:14 2020

@author: Thomas
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from helferlein import files, histplot, intensity2height


# pfad zum ordner wo die csv datein liegen
path = r'D:\data science\fires-from-space-australia-and-new-zeland'

# reinladen der einzelnen datensätze und speichern als dataframe
m6_archive  = pd.read_csv(path+os.sep+files(path)[0]).drop('type', 1)
v1_archive  = pd.read_csv(path+os.sep+files(path)[1]).drop('type', 1)
m6_nrt      = pd.read_csv(path+os.sep+files(path)[2])
v1_nrt      = pd.read_csv(path+os.sep+files(path)[3])

v1_archive['daynight'] = m6_archive['daynight']

v1_archive = v1_archive.rename(columns={"bright_ti4": "brightness"})
v1_nrt = v1_nrt.rename(columns = {'bright_ti4': 'brightness'})

# kombinieren von den 4 dataframes zu einer liste
dframes = [m6_archive, v1_archive, m6_nrt, v1_nrt]

# =============================================================================
# fig, ax = plt.subplots(2,2, num = 'locations', figsize = (16,12))
# a = ax.ravel()
# for i,j in enumerate(dframes):
#     
#     bla = a[i].scatter(j.longitude, j.latitude, marker = '.', label = files(path)[i], c = j.brightness-273.15)
#     fig.colorbar(bla, ax = a[i])
#     a[i].legend()
# 
# fig, ax = plt.subplots(2,2, num = 'hist brighness', figsize = (16,12))
# a = ax.ravel()
# for i,j in enumerate(dframes):
#     
#     a[i].hist(j.brightness.values-273.15, bins = 'auto', density = True, label = files(path)[i])
#     a[i].legend()
# 
# dropfeatures = [0,1,3,4,5,6,7,8,9,10,13]
# 
# for k,l in enumerate(dframes):
#     
#     fig, ax = plt.subplots(2,2, num = files(path)[k], figsize = (16,12))
#     a = ax.ravel()
#     for i,j in enumerate(l.drop(l.columns[dropfeatures], axis = 1)):
#         
#         bla = a[i].scatter(l.longitude, l.latitude, marker = '.', label = j, c = l[j])
#         fig.colorbar(bla, ax = a[i])
#         a[i].legend()
#         if i == 3: continue
# =============================================================================

# offset damit die punkte auf der Karte bleiben XD
offset = 20
# strecken und verschieben der longitude und latitude Werte
longinew = intensity2height(m6_archive['longitude'], data_dic13['TempMax'][40].shape[1]-offset+15, offset+10)
latinew = intensity2height(m6_archive['latitude'], offset,data_dic13['TempMax'][40].shape[0]-offset)

# imcolor ist meine funktion für plotten von bildern in falschfarben
# scatter verteilt die punkte auf dem bild
implot(data_dic13['TempMax'][40], 'tmax w40')
sns.kdeplot(longinew, latinew, cmap="Reds", shade=True, cut = 0)


imcolor(data_dic13['TempMax'][40], 'tmax w40')
plt.scatter(longinew, latinew, label = 'm6_archive brightness', c = m6_archive['brightness'], linewidths=2, alpha = .5, cmap = 'jet')
# maske für die tage in der ersten august woche
mask = (m6_archive['acq_date'] >= '2019-08-01') & (m6_archive['acq_date'] < '2019-08-05')
# maske angewandt auf das gesamte dataframe -> nur die tage der ersten aug woche (= wo mask true ist) werden herausgefiltert
kw31 = m6_archive.loc[mask]


# hier wird die 2,3 und 4 aug woche und die 4 sep wochen als maske in die jeweiligen listen kwlist_aug und kwlist_sep gepeichert

kwlist_aug = []
kwlist_sep = []

for i in range(3):
    start = str(5+7*i)
    if len(start) == 1: start = '0'+start
    end = str(12+7*i)
    kwlist_aug.append((m6_archive['acq_date'] >= '2019-08-{}'.format(start)) & (m6_archive['acq_date'] < '2019-08-{}'.format(end)))
    
for i in range(4):
    start = str(2+7*i)
    if len(start) == 1: start = '0'+start
    end = str(9+7*i)
    if end == '30': break
    kwlist_sep.append((m6_archive['acq_date'] >= '2019-09-{}'.format(start)) & (m6_archive['acq_date'] < '2019-09-{}'.format(end)))
    

# in der for-schleife gehe ich die 3 aug wochen durch und plotte das 'tempmax' bild mit namen 'twmax w32/33/34' 
# damit es immer ein neues fenster mit neuem Namen macht
# im scatter befehl kannst du bei c die eigenschaft ändern (statt brightness zB frp), alpha ist ein wert zwischen
# 0 und 1 für die transparenz der punkte, edgecolor ist die farbe der umrandung der punkte, linewidths ist die größe der punkte
for i in range(3):
    imcolor(data_dic13['TempMax'][32+i], 'tmax w{}'.format(str(32+i)))
    plt.scatter(longinew[kwlist_aug[i]], latinew[kwlist_aug[i]], 
                label = 'm6_archive brightness', c = m6_archive.loc[kwlist_aug[i]]['brightness'], linewidths=2, alpha = .5, edgecolor = 'r')

# hier wird für zB frp der mean, std, max, min, usw angegeben für jede woche
for i in range(3): print(m6_archive.loc[kwlist_aug[i]]['frp'].describe())


















