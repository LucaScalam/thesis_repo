#!/usr/bin/env python
# coding: utf-8

# ### This notebook shows a possible way of creating new points to train, making use of the segmentated fields from previous work

# In[11]:


import geopandas as gpd
import numpy as np
import pandas as pd
from tsaug import TimeWarp
import ee
from sentinelhub import CRS
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

ee.Initialize()
pd.options.mode.chained_assignment = None


# In[2]:


def get_data(img): #function to map 
    date = ee.Date(img.get('system:time_start')).format('D')     
    value = img.reduceRegion(ee.Reducer.mean(), p_get,10,bestEffort = True)   
    B2 = value.get('B2')
    B3 = value.get( 'B3')
    B4 = value.get( 'B4')
    B8 = value.get( 'B8')
    B11 = value.get( 'B11')
    B12 = value.get( 'B12')
    QA60 = value.get( 'QA60')
    ft = ee.Feature(None, {'date': date, \
                          'B2': B2,
                          'B3': B3,
                          'B4': B4,
                          'B8': B8,
                          'B11': B11,
                          'B12': B12,
                          'QA60': QA60,
                          })
    return ft

def get_pd(ft): #pandas dataset
    ft = ft.getInfo()['features']
    dum = pd.DataFrame(ft [:]).properties
    asd = []
    for i in range(len(ft)):
        asd.append(dum[i])
    asd = pd.DataFrame(asd)
    asd['date'] = asd['date'].apply(pd.to_numeric)
    return asd

def get_X(pdf): #replace cloudy samples and interpolates
    for i in ['B2', 'B3', 'B4','B8', 'B11', 'B12']:
        pdf.loc[(pdf.QA60 != 0),i]=np.NaN  #replace cloudy days with NaN
    pdf = pdf[['date','B2', 'B3', 'B4', 'B8', 'B11', 'B12']]

    for i in ['B2', 'B3', 'B4','B8', 'B11', 'B12']:
        dum = pdf[i]
        pdf[i] = dum.interpolate().fillna(method='bfill').values.ravel() #interpolates NaN
    
    pdf.date = pdf.date//5
    pdf = pdf.drop_duplicates(subset=['date'])  #removes duplicates from Sentintel 2 A/B
    pdf = pdf.drop('date',axis=1)
    pdf = pdf.iloc[:48]
    X = pdf.values
    return X


# In[3]:


df = pd.read_csv('../data/nogit_data/points_with_eopatch_4326_test.csv')
df = gpd.GeoDataFrame(df)

for i in range(len(df)):
    if type(df['geo_segmentated_field'].iloc[i]) == float:
        df['geo_segmentated_field'].iloc[i] = None
       
df['geo_segmentated_field'] = gpd.GeoSeries.from_wkt(df['geo_segmentated_field'])
df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
df = df.drop(columns='Unnamed: 0')
df = df.reset_index()


# In[4]:


dataf_1920 = df.loc[df['Campania'] == '19/20']
dataf_1920 = dataf_1920.reset_index()

dataf_1819 = df.loc[df['Campania'] == '18/19']
dataf_1819 = dataf_1819.reset_index()

X = [] #create empty arrays
Y = []
ee.Initialize()


# In[5]:


#campaign 18/19
for l in range(len(dataf_1819)):
    print(len(dataf_1819) - l)
    if(dataf_1819.geo_segmentated_field.iloc[l] == None or dataf_1819.geo_segmentated_field.iloc[l].area >= 1e-4):
        p = ee.Geometry.Point(dataf_1819.iloc[l].Longitud, dataf_1819.iloc[l].Latitud).buffer(100)
    else:
        x,y = dataf_1819.geo_segmentated_field.iloc[l].exterior.xy
        aux = list(zip(x,y))
        p = ee.Geometry.Polygon(aux).buffer(2)
    S2_collection = ee.ImageCollection("COPERNICUS/S2").filterBounds(p).filterDate('2018-09-01', '2019-04-30')                         .sort('system:id', opt_ascending=True)                                                 
    p_get = p                 
    featCol = S2_collection.map(get_data)
    pdf = get_pd(featCol)
    X.append(get_X(pdf))
    Y.append(dataf_1819.iloc[l].Cultivo)


# In[6]:


#campaign 19/20
for l in range(len(dataf_1920)):
    print(len(dataf_1920) - l)
    if(dataf_1920.geo_segmentated_field.iloc[l] == None or dataf_1920.geo_segmentated_field.iloc[l].area >= 1e-4):
        p = ee.Geometry.Point(dataf_1920.iloc[l].Longitud, dataf_1920.iloc[l].Latitud).buffer(100)
    else:
        x,y = dataf_1920.geo_segmentated_field.iloc[l].exterior.xy
        print(dataf_1920.geo_segmentated_field.iloc[l].area)
        aux = list(zip(x,y))
        p = ee.Geometry.Polygon(aux).buffer(2)
    S2_collection = ee.ImageCollection("COPERNICUS/S2").filterBounds(p).filterDate('2019-09-01', '2020-04-30')                         .sort('system:id', opt_ascending=True) 
    p_get = p                 
    featCol = S2_collection.map(get_data)
    pdf = get_pd(featCol)
    X.append(get_X(pdf))
    Y.append(dataf_1920.iloc[l].Cultivo)


# In[7]:


X = np.asarray(X)      
Y = np.asarray(Y)


# In[12]:


np.save('../data/x_test_3', X) #save
np.save('../data/y_test_3', Y)   