# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:10:53 2020

@author: Edgar Alfred Johnson
"""


#wetterimport


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



weather_archive = pd.read_csv(r"D:\Dokumente\PythonStuff\fires-from-space-australia-and-new-zeland\weatherAUS.csv")
globaltemp = pd.read_csv(r"D:\Dokumente\PythonStuff\climate-change-earth-surface-temperature-data\GlobalTemperatures.csv")
temp_countrys = pd.read_csv(r"D:\Dokumente\PythonStuff\climate-change-earth-surface-temperature-data\GlobalLandTemperaturesByCountry.csv")
rain = pd.read_csv(r"D:\Dokumente\PythonStuff\wettercsvseit2006\rain.csv")
wind = pd.read_csv(r"D:\Dokumente\PythonStuff\wettercsvseit2006\wind.csv")
lightning = pd.read_csv(r"D:\Dokumente\PythonStuff\wettercsvseit2006\lightning.csv")

rain.keys()
wind.keys()
lightning.keys()

rain.drop('Unnamed: 16' , 1)

weather_archive.describe()
weather_archive.keys()

globaltemp.describe()
globaltemp.keys()

temp_countrys.describe()
temp_countrys.keys()

temp_countrys['Country']


tempaustralia = temp_countrys.groupby([temp_countrys.Country == 'Australia'])

plt.plot(tempaustralia)
