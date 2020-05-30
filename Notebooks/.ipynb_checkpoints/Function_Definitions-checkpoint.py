### Function Definitions//All Functions and Librarys Used For Project ###
## Schoeggl_Vorauer 2020 ##

import pandas as pd
import numpy as np
import re
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import swifter # swifter is for multiprocessing pandas apply functions

def datetime(df):
    '''spalte acq_date als Datetime auszeichnen'''
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    return df 

def timeselect(df, freq = 'D', fct = 'mean', columns = ['frp','brightness', 'bright_t31']):
    df['acq_date'] = pd.to_datetime(df['acq_date'])-pd.to_timedelta(7, unit='d')
    if fct == 'mean': out = df.groupby([pd.Grouper(key='acq_date', freq=freq)])[columns].mean().reset_index().sort_values('acq_date')
    if fct == 'max':  out = df.groupby([pd.Grouper(key='acq_date', freq=freq)])[columns].max().reset_index().sort_values('acq_date')
    if fct == 'min':  out = df.groupby([pd.Grouper(key='acq_date', freq=freq)])[columns].min().reset_index().sort_values('acq_date')
    return out

def date_separating(df, sel): 
    '''splits dataframes into weeks, months, years'''
    out = []
    if 'D' in sel:
        out.append([g.reset_index() for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'D'))])
    if 'W' in sel:
        out.append([g.reset_index() for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'W'))])
    if 'M' in sel:
        out.append([g.reset_index() for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'M'))])
    if 'Y' in sel:
        out.append([g.reset_index() for n, g in df.set_index('acq_date').groupby(pd.Grouper(freq = 'Y'))])
    return out

def get_state(df):
    def conds(row):
        x, y = row['latitude'], row['longitude']
        a = np.array([[141,150],[1,1]])
        b = np.array([-34,-37.5])
        (k,d) = np.linalg.solve(a,b)

        if y < -129:
            return 'WA'
        if (x > -26) & (129 <= y < 138):
            return 'NT'
        if (x <= -26) & (129 <= y < 141):
            return 'SA'
        if (x > -26) & (138 <= y < 141):
            return 'QLD'
        if (x > -29) & (141 >= y):
            return 'QLD'
        if (-34 < x <= -29) & (y >= 141):
            return 'NSW'
        if (x+k*y > d) & (y >= 141):
            return 'NSW'
        if (x+k*y <= d) & (y >= 141):
            return 'VIC'
        if (-39.2 <= x < 37.5) & (y >= 141):
            return 'VIC'
        return 'TAS'
    df['State'] = df.swifter.apply(conds, axis=1)
    return df

def stringchange(df, column = 'Location'):
    
    result = []
    for i in df[column]:
        sep = ' '
        result.append(sep.join(re.findall('[A-Z][^A-Z]*', i )))
    df[column] = result
    return df

def geolookup(row):
    gc = Nominatim(user_agent="fintu-blog-geocoding-python")
    #applying geocoordinates to locations
    def rate_limited_geocode(query):
        geocode = RateLimiter(gc.geocode, min_delay_seconds=1)
        return geocode(query)
    #geocode der 2ten funktion ist eine andere variable
    def geocodeb(row):
        lookup_query = row["Location"] + ", " + "Australia"
        lookup_result = rate_limited_geocode(lookup_query)
        return lookup_result
    return geocodeb(row)
#test: weatherloc["loctemp"] = weatherloc.apply(geocodeb, axis=1)

'''filling function for longitude, latitude into correct rows'''
def reverse_filling(df1, df2):
    templatitude = np.zeros(df1.shape[0])
    templongitude = np.zeros(df1.shape[0])
    for i in df1.Location.unique():
        templatitude[df1.Location == i] = df2[df2.Location==i].latitude
        templongitude[df1.Location == i] = df2[df2.Location==i].longitude
    df1['latitude'] = templatitude
    df1['longitude'] = templongitude
    return df1

def year_month(df):
    year = np.array([(d.year, d.month) for d in df['acq_date']])
    df['Year'],df['Month'] = year.T 
    return df