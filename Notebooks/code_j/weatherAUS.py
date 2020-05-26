# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:44:10 2020

@author: Edgar Alfred Johnson
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

dfweather_full = pd.read_csv(r'D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\weatherAUS.csv')
df_climate_by_country = pd.read_csv(r'D:\Dokumente\PythonStuff\climate-change-earth-surface-temperature-data\GlobalLandTemperaturesByCountry.csv')



dfweather_full.shape

dfweather_full.head()

print(dfweather_full.keys())


print(dfweather_full.describe())

print(dfweather_full.corr())

print(dfweather_full['Date'])

print(df_climate_by_country.keys())

print(df_climate_by_country['dt'].describe())

print(df_climate_by_country.describe())
    
df_australia = df_climate_by_country[df_climate_by_country.Country == 'Australia']

print(df_australia.describe())