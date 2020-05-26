# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:34:02 2020

@author: Edgar Alfred Johnson as Johanna Schoeggl
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from pandas import Timestamp
import csv
from helferlein_thomas import files
#from firesfromSpace import date_seperating 
from scipy.stats import poisson
from scipy.stats import expon
import folium
from IPython.display import HTML, display

#alle nasadaten von 2000 bis 2020

# pfad zum ordner wo die csv datein liegen
path = r'D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\allnasa'

# reinladen der einzelnen datensätze und speichern als dataframe
m6_archive  = pd.read_csv(path+os.sep+files(path)[0]).drop('type', 1)
v1_archive  = pd.read_csv(path+os.sep+files(path)[1]).drop('type', 1)
m6_nrt      = pd.read_csv(path+os.sep+files(path)[2])
v1_nrt      = pd.read_csv(path+os.sep+files(path)[3])

#damit die verteilungen sinn ergebe sollte man vermutlich die unwichtigen spalten droppen!  
v1a_renamed = v1_archive.rename(columns={'bright_ti4':'brightnessV'})
v1n_renamed = v1_nrt.rename(columns={'bright_ti4':'brightnessV'})

V1a = v1a_renamed.copy()
V1n = v1n_renamed.copy()
M6a = m6_archive.copy()
M6n = m6_nrt.copy()

M6a.drop(['acq_time', 'satellite', 'instrument', 'version', 'daynight'], 1 )



M6a['acq_date'] = pd.to_datetime(M6a['acq_date']) - pd.to_timedelta(7, unit='d')

df = M6a.groupby([pd.Grouper(key='acq_date', freq='W-MON')])['brightness', 'bright_t31'].mean().reset_index().sort_values('acq_date')

#plt.plot(np.array(df.brightness))
plt.figure()
plt.plot(df.brightness)
plt.plot(df.bright_t31)


V1a['acq_date'] = pd.to_datetime(V1a['acq_date']) - pd.to_timedelta(7, unit='d')

df2 = V1a.groupby([pd.Grouper(key='acq_date', freq='W-MON')])['brightnessV', 'bright_ti5'].mean().reset_index().sort_values('acq_date')

plt.plot(df2.brightnessV)
plt.plot(df2.bright_ti5)  
         
plt.axes()

V1a.keys()
m6_nrt['acq_date'] = pd.to_datetime(m6_nrt['acq_date'])-pd.to_timedelta(7, unit='d')
m6_archive['acq_date'] = pd.to_datetime(m6_archive['acq_date'])-pd.to_timedelta(7, unit='d')
def date_seperating(df, sel): 
    '''splits dataframes into weeks, months, years'''
    out = []
    if 'D' in sel:
        out.append([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'D'))])
    if 'W' in sel:
        out.append([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))])
    if 'M' in sel:
        out.append([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))])
    if 'Y' in sel:
        out.append([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'Y'))])
    return out    
    # return ([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'D'))],
    #         [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))], 
    #         [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))],
    #         [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'Y'))])

weeks, month, year = date_seperating(m6_archive)
week0 = weeks[0]

day_test =[g for n, g in m6_nrt.set_index('acq_date').groupby(pd.Grouper(freq = 'D'))]
daym =[g for n, g in m6_archive.set_index('acq_date').groupby(pd.Grouper(freq = 'D'))]
# m6_apoisson = poisson.rvs(mu=3, size=df)

# #poissonverteilung modis archive

# ax = sns.distplot(m6_apoisson,
#                   bins=30,
#                   kde=False,
#                   color='orange',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Poisson Distribution', ylabel='Brightness')

# #[Text(0,0.5,u'Brightness'), Text(0.5,0,u'Brightness Distribution')]

# v1_apoission = v1_archive.copy()
# v1_apoission.drop(['acq_time', 'satellite', 'instrument', 'version'], 1)

# #plt.figure()

# v1_apoisson = poisson.rvs(mu=3, size=10000)

# #poissonverteilung viriis archive

# ax = sns.distplot(v1_apoisson,
#                   bins=30,
#                   kde=False,
#                   color='orange',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Poisson Distribution', ylabel='Brightness')

# #[Text(0,0.5,u'Brightness'), Text(0.5,0,u'Brightness Distribution')]

# #exponetial distr



# m6_aexp = m6_archive.copy()
# m6_aexp.drop(['acq_time', 'satellite', 'instrument', 'version', 'daynight'], 1 )
# v1_aexp = m6_archive.copy()
# v1_aexp.drop(['acq_time', 'satellite', 'instrument', 'version'], 1)
# #from scipy.stats import expon

# m6_aexp = expon.rvs(scale=1,loc=0,size=1000)
# ax = sns.distplot(m6_aexp,
#                   kde=True,
#                   bins=100,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Exponential Distribution', ylabel='Brightness')

# #[Text(0,0.5,u'Brightness'), Text(0.5,0,u'Exponential Distribution')]

# #figure()

# v1_aexp = expon.rvs(scale=1,loc=0,size=1000)
# ax = sns.distplot(v1_aexp,
#                   kde=True,
#                   bins=100,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Exponential Distribution', ylabel='Brightness')

# #[Text(0,0.5,u'Brightness'), Text(0.5,0,u'Exponential Distribution')]

#map(lambda, ['longitude', 'latitude', 'longitude', 'latitude']) = 

#Create a map
f = folium.Figure(width=1000, height=500)
center_lat = -24.003249 
center_long = 133.737310
m = folium.Map(location=[center_lat,center_long], control_scale=True, zoom_start=4,width=750, height=500,zoom_control=True).add_to(f)
for i in range(0,week0.shape[0]):    
    location=[week0.iloc[i]['latitude'], week0.iloc[i]['longitude']]
    folium.CircleMarker(location,radius=1,color='red').add_to(m)

display(m)




#so man muss die parts durch alle i aus week0 ersetzen (alle koordinaten) und dann?

# w = shapefile.Writer(shapefile.POLYGON)
 
# w.poly(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
#  w.field('FIRST_FLD','C','40')
#  w.field('SECOND_FLD','C','40')
#  w.record('First','Polygon')
#  w.save('shapefiles/test/polygon')

#datum + koordinate : feuereuabreitung koordinaten abstand

days = date_seperating(m6_archive, 'D')

# mathode k-means-clustering
from sklearn.cluster import KMeans

X = np.array(days[0][0][['latitude','longitude']])
Y = np.array(days[0][1][['latitude','longitude']])

kmeans = KMeans(n_clusters=15).fit_predict(X)     
kmeans.predict(Y)
kmeans.cluster_centers_
#kmeans.fit(Y)




# for i in range(len(m6_archive)):
#     for j in range(i, len(m6_archive)):
#         np.array([m6_archive['latitude'][i], m6_archive['longitude'][i]]) -
#         np.array([m6_archive['latitude'][j], m6_archive['longitude'][j]])


plt.figure()
plt.scatter(days[0][1].longitude, days[0][1].latitude, c=kmeans)

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

#!!!JS???!!!  Plot the decision boundary. For that, we will assign a color to each cluster !!!JS???!!! 
x_min, x_max = kmeans[0 , 1].min() - 1, kmeans[0 , 1].max() + 1
y_min, y_max = kmeans[0 , 1].min() - 1, kmeans[0 , 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(kmeans[0 , 1], kmeans[0 ,1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering fires')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
   