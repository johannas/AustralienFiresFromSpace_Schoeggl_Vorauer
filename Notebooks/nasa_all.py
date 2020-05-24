# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:13:45 2020

@author: Thomas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import scipy.ndimage as nd
from scipy import fftpack
from skimage.measure import regionprops

from helferlein import imcolor, files


allmodis = pd.read_csv(r'D:\data science\DL_FIRE_M6_126870\fire_archive_M6_126870.csv').drop(['acq_time', 'satellite', 'instrument', 'version', 'daynight', 'type'], 1)
allviirs = pd.read_csv(r'D:\data science\DL_FIRE_V1_126871\fire_archive_V1_126871.csv').drop(['acq_time', 'satellite', 'instrument', 'version', 'type'], 1)

allmodis['acq_date'] = pd.to_datetime(allmodis['acq_date']) - pd.to_timedelta(7, unit='d')
allviirs['acq_date'] = pd.to_datetime(allviirs['acq_date']) - pd.to_timedelta(7, unit='d')

# allmodis['acq_date'] = pd.to_datetime(allmodis['acq_date']) - pd.to_timedelta(7, unit='d')
# df = allmodis.groupby([pd.Grouper(key='acq_date', freq='W-MON')])['frp'].mean().reset_index().sort_values('acq_date')

def wmy(df):
    return ([g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))],
            [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))],
            [g for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'Y'))])

# weeks   = [g for n, g in allmodis.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))]
# months  = [g for n, g in allmodis.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))]
# years   = [g for n, g in allmodis.set_index('acq_date').groupby(pd.Grouper(freq = 'Y'))]

weeks_m, months_m, years_m = wmy(allmodis)
weeks_v, months_v, years_v = wmy(allviirs)

ym1 = years_m[1].groupby(pd.Grouper(freq = 'M')).mean()
ym2 = years_m[2].groupby(pd.Grouper(freq = 'M')).mean()

ym = [i.groupby(pd.Grouper(freq = 'M')).mean() for i in years_m]


plt.figure('brightness')
plt.clf()
for i in range(15,20):
    plt.plot(ym[i].reset_index().brightness, 'o-', label = 'brightness-{}'.format(i))
# plt.plot(ym2.reset_index().brightness, 'o-', label = 'brightness2')
plt.legend()
























