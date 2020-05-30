# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:19:05 2020

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



dfM6_archive = pd.read_csv(r"D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\fire_archive_M6_96619.csv")
dfM6_nrt = pd.read_csv(r"D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\fire_nrt_M6_96619.csv")
dfV1_archive = pd.read_csv(r"D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\fire_archive_V1_96617.csv")
dfV1_nrt = pd.read_csv(r"D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\fire_nrt_V1_96617.csv")


dfM6_archive.head()


dfM6_archive.shape

print(dfM6_archive.keys())
print(dfV1_archive.keys())
print(dfV1_nrt.keys())
print(dfM6_nrt.keys())
#ti's = sensoren
plt.hist(dfM6_archive.bright_t31, bins=100 , alpha=0.4 , label = 'bright_t31')
plt.hist(dfM6_archive.brightness, bins=100 , alpha=0.4 , label = 'brightness')

plt.hist(dfV1_archive.bright_ti4, bins=100 , alpha=0.4 , label = 'bright_ti4')
plt.hist(dfV1_archive.bright_ti5, bins=100 , alpha=0.4 , label = 'bright_ti5')

plt.legend()

plt.figure()    
     
plt.hist(dfM6_nrt.bright_t31, bins=100 , alpha=0.4 , label = 'bright_t31')
plt.hist(dfM6_nrt.brightness, bins=100 , alpha=0.4 , label = 'brightness')

plt.hist(dfV1_nrt.bright_ti4, bins=100 , alpha=0.4 , label = 'bright_ti4')
plt.hist(dfV1_nrt.bright_ti5, bins=100 , alpha=0.4 , label = 'bright_ti5')

plt.legend()


print(dfM6_archive.max())

df_haeufigster = dfM6_archive['bright_t31'].value_counts().max()
print(df_haeufigster)
df_counts = df_haeufigster = dfM6_archive['bright_t31'].value_counts()
print(df_counts)

dfM6_archive['bright_t31'].describe()

plt.figure()
plt.plot(dfM6_archive['acq_date'], dfM6_archive['brightness'])

print(dfM6_archive['acq_date'])
dfM6_archive.groupby(['acq_date']).describe()
plt.figure()
plt.plot(dfM6_archive.groupby(['acq_date']).brightness.mean())

plt.figure()
plt.plot(dfV1_archive.groupby(['acq_date']).bright_ti4.mean())

plt.figure()
plt.plot(dfM6_nrt.groupby(['acq_date']).brightness.mean())

plt.figure()
plt.plot(dfV1_nrt.groupby(['acq_date']).bright_ti4.mean())

plt.figure()
plt.plot(dfV1_nrt.groupby(['acq_date']).bright_ti5.mean())

plt.figure()
sns.scatterplot(dfM6_archive['latitude'], dfM6_archive['longitude'])


plt.figure()
dfM6_archive.corr()
plt.figure()
dfV1_archive.corr()

print(dfM6_archive['acq_date'])
print(dfV1_archive['acq_date'])

df_cor_V1A_M6A = dfV1_archive.corrwith(dfM6_archive)

df_cor_V1N_M6N = dfV1_nrt.corrwith(dfM6_nrt)

df_Mcor_V1A_M6N = dfV1_archive.corrwith(dfM6_nrt)

df_Mcor_V1N_M6A = dfV1_nrt.corrwith(dfM6_archive)

print(df_cor_V1A_M6A)
print(df_cor_V1N_M6N)
print(df_Mcor_V1A_M6N)
print(df_Mcor_V1N_M6A)

# dfM6_archive.groupby("acq_date, brightness").mean()

#import dabl
#dabl.plot(data, target_col='type', type_hints={'type': 'categorical'})

dfM6_archive['acq_date'].max()
dfV1_archive['acq_date'].max()
dfM6_nrt['acq_date'].max()
dfV1_nrt['acq_date'].max()


dfM6_archive.groupby( ["brightness","acq_date"]).count()

dfMixed_M = pd.concat([dfM6_archive, dfM6_nrt], axis=1)
print(dfMixed_M.shape)
print(dfM6_archive.shape)
print(dfV1_archive.shape)
print(dfM6_nrt.shape)
print(dfV1_nrt.shape)
dfMixed_M2 = pd.concat([dfM6_archive, dfM6_nrt], ignore_index =True)
print(dfMixed_M2.shape)
dfMixed_V = pd.concat([dfV1_archive, dfV1_nrt], ignore_index =True)
print(dfMixed_V.shape)
df_cor_Mmix_Vmix = dfMixed_M2.corrwith(dfMixed_V)
print(df_cor_Mmix_Vmix)


plt.plot(dfMixed_M2.groupby(['acq_date']).brightness.mean())
plt.plot(dfMixed_M2.groupby(['acq_date']).bright_t31.mean())

dfMixed_V.keys()
plt.plot(dfMixed_V.groupby(['acq_date']).bright_ti4.mean())
plt.plot(dfMixed_V.groupby(['acq_date']).bright_ti5.mean())

plt.figure()
px.bar(dfMixed_M2, x = 'acq_date', y = 'brightness', orientation = 'v')

plt.figure()
px.bar(dfMixed_V, x = 'acq_date', y = 'bright_ti4', orientation = 'v')

correlation_matrix = dfMixed_M2.corr() 
print(correlation_matrix)

correlation_matrix2 = dfMixed_V.corr() 
print(correlation_matrix2)

#dfM1_archive_fig = go.Figure(data=go.Heatmap(x=correlation_matrix.columns, y=correlation_matrix.columns,z=correlation_matrix))
#fig.show()

print(dfMixed_M.info())
print(dfMixed_M.isnull().values.sum())
print(dfMixed_M2.isna().sum())

#fig = px.line(dfMixed_V1[dfMixed_V1.bright_ti4 == 'longitude'], x="Date", y="bright_ti4")



X0= dfMixed_M2.drop('acq_date', axis=1)
y0 = dfM6_nrt.drop('acq_date', axis=1)
#X = X0.drop('brightness', axis=1)
#y = y0.brightness

#print(X.shape)
#print(y.shape)
#X= X.transpose()
#print(X.shape)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#regr = RandomForestRegressor()
#regr.fit(X_train, y_train)

#y_pred = regr.predict(X_test)
#print(y_pred)

plt.plot(dfMixed_M2.groupby(['longitude']).brightness.mean())
plt.plot(dfMixed_M2.groupby(['longitude']).bright_t31.mean())

dfMixed_V.keys()
plt.plot(dfMixed_V.groupby(['longitude']).bright_ti4.mean())
plt.plot(dfMixed_V.groupby(['longitude']).bright_ti5.mean())

plt.figure()
sns.scatterplot(dfM6_archive.longitude, dfM6_archive.latitude, hue= dfM6_archive.brightness)

plt.figure()
sns.scatterplot(dfM6_archive.longitude, dfM6_archive.latitude, hue= dfM6_archive.bright_t31)


plt.figure()
sns.scatterplot(dfV1_archive.longitude, dfV1_archive.latitude, hue= dfV1_archive.bright_ti4)

plt.figure()
sns.scatterplot(dfV1_archive.longitude, dfV1_archive.latitude, hue= dfV1_archive.bright_ti5)


df_M6A_renamed = dfM6_archive
df_M6N_renamed = dfM6_nrt

df_V1A_renamed = dfV1_archive.rename(columns={'bright_ti4':'brightnessV'})
df_V1N_renamed = dfV1_nrt.rename(columns={'bright_ti4':'brightnessV'})



dframes = [df_M6A_renamed, df_V1A_renamed, df_M6N_renamed, df_V1N_renamed]
dfnames = ["df_M6A_renamed", "df_V1A_renamed", "df_M6N_renamed", "df_V1N_renamed"]

fig, ax = plt.subplots(2,2, num = 'locations', figsize = (16,10))
a = ax.ravel()
for i,j in enumerate(dframes):
#ae = achsenelement  
    ae = a[i].scatter(j.longitude, j.latitude, marker = '.', label = dfnames[i] , c = j.frp)
    fig.colorbar(ae, ax = a[i])
    a[i].legend()
    a[i].grid(True)

plt.show()



# X1= dfM6_archive.drop('type', axis=1)
# y1 = dfM6_nrt.drop('type', axis=1)
# X = X1.drop('brightness', axis=1)
# y = y1.brightness

# print(X.shape)
# print(y.shape)
# X= X.transpose()
# print(X.shape)

# X_train, X_test, y_train, y_test = train_test_split(
#      X, y, test_size=0.33, random_state=42)


# #regr = Regressionclassifier()
# #regr.fit(X_train, y_train)

# #y_pred = regr.predict(X_test)
# #print(y_pred)



# clas = RandomForestClassifier()
# clas.fit(X_train, y_train)

# y_pred = clas.predict(X_test)
# print(y_pred)
#%%
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.hist(data.brightnessV,bins='auto');
# ax.set_title("Histogram of Brightness Temperatures I4");
# ax.set_xlabel("Brightness Temperature I4");
# ax.set_ylabel("Counts");

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.hist(data.brightnessV,bins=200);
# ax.set_xlim(290,360);
# ax.set_ylim(5000,1.55e4);
# ax.set_title("Histogram of Brightness Temperatures I4");
# ax.set_xlabel("Brightness Temperature I4");
# ax.set_ylabel("Counts");

# #Assuming Saturated brightness are those that are above 330, based on the above histogram

# mask = data['brightnessV'] >= 330
# sat_df = data[mask]
# unsat_df = data[~mask]

# fig, axes = plt.subplots(1,2, figsize =(20,10), subplot_kw={'xticks': (), 'yticks': ()})
# img = axes[0].scatter(sat_df.longitude,sat_df.latitude, alpha=0.01, color='red')
# axes[0].set_title("Scatter map for Saturated Brightness I4")
# img = axes[1].scatter(unsat_df.longitude,unsat_df.latitude, alpha=0.01)
# axes[1].set_title("Scatter map for UnSaturated Brightness I4")

# for ax in axes.ravel():
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
# plt.tight_layout()
# #fig.colorbar(img, ax=axes[1])
# plt.show()

# fig, axes = plt.subplots(1,2, figsize =(20,10), subplot_kw={'xticks': (), 'yticks': ()})
# img = axes[0].scatter(sat_df.longitude,sat_df.latitude, alpha=0.008, color='red')
# axes[0].set_title("Scatter map for Saturated Brightness I4")
# img = axes[1].scatter(unsat_df.longitude,unsat_df.latitude, alpha=0.008)
# axes[1].set_title("Scatter map for UnSaturated Brightness I4")

# for ax in axes.ravel():
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
# plt.tight_layout()
# #fig.colorbar(img, ax=axes[1])
# plt.show()
#%%



#im osten brennen die punkte länger (punkte - feuerpixel - pro poisition date tag)
dfM6_archive.groupby['longitude' , 'latitude', 'acq_date'].brightness.describe
#feuerhäufung ort und zeit + feuer true false: nimmt er immer jeden 

#wann  kommt ein neuer feuerpixel dazu, kurzer zetlicher abstand : ausbreitung, länger: kann zur feuerhäufung genommen werden
#corr matrix zw. feuer pro zeit und ort.
  


#%%

dfteil2019 = dfM6_archive[dfM6_archive.acq_date == '2019-08-22'].brightness
print(dfteil2019)


#dfhottestWeek

dfM6_archive_dated = dfM6_archive.copy()

dfM6_archive_dated['acq_date'] = pd.to_datetime(dfM6_archive_dated['acq_date']) - pd.to_timedelta(7, unit='d')


def date_seperating(df): 
    '''splits dataframes into weeks, months, years'''
    return ([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))], 
            [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))],
            [g for n, g in df.set_index ('acq_date').groupby(pd.Grouper(freq = 'Y'))])


weeks = [g for n, g in dfM6_archive_dated.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))]
months = [g for n, g in dfM6_archive_dated.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))]
#years = [g for n, g in dfM6_archive_dated.set_index ('acq_date').groupby(pd.Grouper(freq = 'Y'))]

weeks[0].describe()
weeks_array0 = [i.brightness.max() for i in weeks]
weeks_array1 = [i.groupby(['longitude', 'latitude']).brightness.max() for i in weeks]
weeks_array2 = [i[i.brightness == i.brightness.max()] for i in weeks]
#week1 und 2 contains dataframes



# print(weeks_array0, weeks_array1, weeks_array2)

# for i, j in enumerate([weeks_array1, weeks_array2]):
#     for a, b in enumerate(j):
#         b.to_csv(r'D:\Dokumente\PythonStuff\csvExport\week{}{}.csv'.format(i,a))

# for i, j in enumerate([weeks_array0, weeks_array1, weeks_array2]):
#     with open("D:\Dokumente\PythonStuff\csvExport\week{}.csv".format(i), "w") as f:
#         wr = csv.writer(f, delimiter='\n')
#         wr.writerow(j)

#503K 2019-08-22 (koordinate x, y)
print(np.argmax(weeks_array0))

#picklemethode verwenden!
        
#weeks = [g for n, g in dfM6_archive.groupby(pd.Grouper(key='acq_date',freq='W'))]
#months = [g for n, g in dfM6_archive.groupby(pd.Grouper(key='acq_date',freq='M'))]

#weeks = [g for n, g in dfM6_archive.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))]
#months = [g for n, g in dfM6_archive.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))]

corrmat = dfM6_archive.corr()
plt.matshow(corrmat)

corrM6AN = dfM6_archive.corrwith(dfV1_archive)
print(corrM6AN)

coordsys = dfM6_archive[['latitude', 'longitude']]

coordsys.to_csv(r'D:\Dokumente\PythonStuff\csvExport\coordinates.csv', index = False)

