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
from shapely.geometry import Polygon, Point
import random

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

def get_pd(ft): #convert to pandas dataset
    ft = ft.getInfo()['features']
    dum = pd.DataFrame(ft [:]).properties
    asd = []
    for i in range(len(ft)):
        asd.append(dum[i])
    asd = pd.DataFrame(asd)
    asd['date'] = asd['date'].apply(pd.to_numeric)
    return asd

def get_X(pdf): #removes cloudy samples and interpolates them
    for i in ['B2', 'B3', 'B4','B8', 'B11', 'B12']:
        pdf.loc[(pdf.QA60 != 0),i]=np.NaN  #replace cloudy days (qa60 bitmask not 0) with NaN
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

def get_random_point_in_polygon(poly):
     minx, miny, maxx, maxy = poly.bounds
     while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             return p

# In[3]:


df = pd.read_csv('../data/nogit_data/points_with_eopatch_4326.csv')
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

N = 10 #number of augmented samples per real sample
augmenter = (
    TimeWarp(n_speed_change = 2, max_speed_ratio=1.5) * N # perform 10 time warp augmentations
)

X = [] 
Y = []
ee.Initialize()

num_points = 10

#campaign 18/19
for l in range(len(dataf_1819)):
    points = []
    print(len(dataf_1819) - l)
    if(dataf_1819.geo_segmentated_field.iloc[l] == None or dataf_1819.geo_segmentated_field.iloc[l].area >= 5e-5):
        points.append(ee.Geometry.Point(dataf_1819.iloc[l].Longitud, dataf_1819.iloc[l].Latitud).buffer(100))
        print('Simple point')
    else:
        points.append(ee.Geometry.Point(dataf_1819.iloc[l].Longitud, dataf_1819.iloc[l].Latitud).buffer(100))
        for j in range(num_points):
            print('get_random: {}'.format(j))
            aux = get_random_point_in_polygon(dataf_1819.geo_segmentated_field.iloc[l])
            points.append(ee.Geometry.Point(aux.bounds[0],aux.bounds[1]).buffer(100))
    if(len(points) > 1):
        for p in points:
            S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                .filterBounds(p).filterDate('2018-09-01', '2019-04-30') \
                .sort('system:id', opt_ascending=True)                                                 
            p_get = p                 
            featCol = S2_collection.map(get_data)
            pdf = get_pd(featCol)
            X.append(get_X(pdf))
            Y.append(dataf_1819.iloc[l].Cultivo)
    else:
        S2_collection = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(points[0]).filterDate('2018-09-01', '2019-04-30') \
            .sort('system:id', opt_ascending=True)                                                 
        p_get = points[0]                 
        featCol = S2_collection.map(get_data)
        pdf = get_pd(featCol)
        X.append(get_X(pdf))
        Y.append(dataf_1819.iloc[l].Cultivo)
        x_aug = augmenter.augment(X[len(X)-1].reshape((1,X[0].shape[0],X[0].shape[1])))
        y_aug = []
        for j in range(N):
            y_aug.append(Y[len(Y)-1]) # create new y array
        y_aug = np.asarray(y_aug)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X2 = np.vstack([X, x_aug]) #join real and augmented arrays
        Y2 = np.hstack([Y, y_aug])
        X = []
        for i in range(X2.shape[0]):
            X.append(X2[i])
        Y = Y2.tolist()
        
            
print('Adding 1920 data in not present classes')
        
not_in_1819 = [5,11,24]
for l in range(len(dataf_1920)):
    print(len(dataf_1920) - l)
    points = []
    if dataf_1920.iloc[l].Cultivo in not_in_1819:
        if(dataf_1920.geo_segmentated_field.iloc[l] == None or dataf_1920.geo_segmentated_field.iloc[l].area >= 5e-5):
            points.append(ee.Geometry.Point(dataf_1920.iloc[l].Longitud, dataf_1920.iloc[l].Latitud).buffer(100))
            print('Simple point')
        else:
            points.append(ee.Geometry.Point(dataf_1920.iloc[l].Longitud, dataf_1920.iloc[l].Latitud).buffer(100))
            for j in range(num_points):
                print('get_random: {}'.format(j))
                aux = get_random_point_in_polygon(dataf_1920.geo_segmentated_field.iloc[l])
                points.append(ee.Geometry.Point(aux.bounds[0],aux.bounds[1]).buffer(100))
        if(len(points) > 1):
            for p in points:
                S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                    .filterBounds(p).filterDate('2019-09-01', '2020-04-30') \
                    .sort('system:id', opt_ascending=True)                                                 
                p_get = p                 
                featCol = S2_collection.map(get_data)
                pdf = get_pd(featCol)
                X.append(get_X(pdf))
                Y.append(dataf_1920.iloc[l].Cultivo)
        else:
            S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                .filterBounds(points[0]).filterDate('2019-09-01', '2020-04-30') \
                .sort('system:id', opt_ascending=True)                                                 
            p_get = points[0]                
            featCol = S2_collection.map(get_data)
            pdf = get_pd(featCol)
            X.append(get_X(pdf))
            Y.append(dataf_1920.iloc[l].Cultivo)
            x_aug = augmenter.augment(X[len(X)-1].reshape((1,X[0].shape[0],X[0].shape[1])))
            y_aug = []
            for j in range(N):
                y_aug.append(Y[len(Y)-1]) # create new y array
            y_aug = np.asarray(y_aug)

            X = np.asarray(X)
            Y = np.asarray(Y)
            X2 = np.vstack([X, x_aug]) #join real and augmented arrays
            Y2 = np.hstack([Y, y_aug])
            X = []
            for i in range(X2.shape[0]):
                X.append(X2[i])
            Y = Y2.tolist()



X = np.asarray(X)
Y = np.asarray(Y)

np.save('../data/x_train_6_1819', X) #save
np.save('../data/y_train_6_1819', Y)

le = LabelEncoder() #create and fit label encoder
le = le.fit(Y.flatten())
y2 = le.transform(Y.flatten()) 

n_classes = y2.max()+1 #14

#scale data using a standard scalar (ie, substract mean and divide by std)
scalers = {}
for i in range(X.shape[1]):
    scalers[i] = StandardScaler()
    scalers[i].fit(X[:, i, :]) 

pickle.dump(scalers, open('../data/scaler_6_1819.pkl','wb')) #save
pickle.dump(le, open('../data/labelencoder_6_1819.pkl','wb'))


print(' ----------------------------------------------------------------------------------------------')
print(' ----------------------------------------------------------------------------------------------')
print('Season 19/20')
print(' ----------------------------------------------------------------------------------------------')
print(' ----------------------------------------------------------------------------------------------')
#campaign 19/20
X = [] 
Y = []
for l in range(len(dataf_1920)):
    points = []
    print(len(dataf_1920) - l)
    if(dataf_1920.geo_segmentated_field.iloc[l] == None or dataf_1920.geo_segmentated_field.iloc[l].area >= 5e-5):
        points.append(ee.Geometry.Point(dataf_1920.iloc[l].Longitud, dataf_1920.iloc[l].Latitud).buffer(100))
        print('Simple point')
    else:
        points.append(ee.Geometry.Point(dataf_1920.iloc[l].Longitud, dataf_1920.iloc[l].Latitud).buffer(100))
        for j in range(num_points):
            print('get_random: {}'.format(j))
            aux = get_random_point_in_polygon(dataf_1920.geo_segmentated_field.iloc[l])
            points.append(ee.Geometry.Point(aux.bounds[0],aux.bounds[1]).buffer(100))
    if(len(points) > 1):
        for p in points:
            S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                .filterBounds(p).filterDate('2019-09-01', '2020-04-30') \
                .sort('system:id', opt_ascending=True)
            p_get = p                 
            featCol = S2_collection.map(get_data)
            pdf = get_pd(featCol)
            X.append(get_X(pdf))
            Y.append(dataf_1920.iloc[l].Cultivo)
    else:
        S2_collection = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(points[0]).filterDate('2019-09-01', '2020-04-30') \
            .sort('system:id', opt_ascending=True)
        p_get = points[0]                
        featCol = S2_collection.map(get_data)
        pdf = get_pd(featCol)
        X.append(get_X(pdf))
        Y.append(dataf_1920.iloc[l].Cultivo)
        x_aug = augmenter.augment(X[len(X)-1].reshape((1,X[0].shape[0],X[0].shape[1])))
        y_aug = []
        for j in range(N):
            y_aug.append(Y[len(Y)-1]) # create new y array
        y_aug = np.asarray(y_aug)

        X = np.asarray(X)
        Y = np.asarray(Y)
        X2 = np.vstack([X, x_aug]) #join real and augmented arrays
        Y2 = np.hstack([Y, y_aug])
        X = []
        for i in range(X2.shape[0]):
            X.append(X2[i])
        Y = Y2.tolist()

print('Adding 1819 data in not present classes')
        
not_in_1920 = [7,8,19]
for l in range(len(dataf_1819)):
    points = []
    if dataf_1819.iloc[l].Cultivo in not_in_1920:
        if(dataf_1819.geo_segmentated_field.iloc[l] == None or dataf_1819.geo_segmentated_field.iloc[l].area >= 5e-5):
            points.append(ee.Geometry.Point(dataf_1819.iloc[l].Longitud, dataf_1819.iloc[l].Latitud).buffer(100))
            print('Simple point')
        else:
            points.append(ee.Geometry.Point(dataf_1819.iloc[l].Longitud, dataf_1819.iloc[l].Latitud).buffer(100))
            for j in range(num_points):
                print('get_random: {}'.format(j))
                aux = get_random_point_in_polygon(dataf_1819.geo_segmentated_field.iloc[l])
                points.append(ee.Geometry.Point(aux.bounds[0],aux.bounds[1]).buffer(100))
        if(len(points)>1):
            for p in points:
                S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                    .filterBounds(p).filterDate('2018-09-01', '2019-04-30') \
                    .sort('system:id', opt_ascending=True)                                                 
                p_get = p                 
                featCol = S2_collection.map(get_data)
                pdf = get_pd(featCol)
                X.append(get_X(pdf))
                Y.append(dataf_1819.iloc[l].Cultivo)
        else:
            S2_collection = ee.ImageCollection("COPERNICUS/S2") \
                .filterBounds(points[0]).filterDate('2018-09-01', '2019-04-30') \
                .sort('system:id', opt_ascending=True)                                                 
            p_get = points[0]                
            featCol = S2_collection.map(get_data)
            pdf = get_pd(featCol)
            X.append(get_X(pdf))
            Y.append(dataf_1819.iloc[l].Cultivo)
            x_aug = augmenter.augment(X[len(X)-1].reshape((1,X[0].shape[0],X[0].shape[1])))
            y_aug = []
            for j in range(N):
                y_aug.append(Y[len(Y)-1]) # create new y array
            y_aug = np.asarray(y_aug)

            X = np.asarray(X)
            Y = np.asarray(Y)
            X2 = np.vstack([X, x_aug]) #join real and augmented arrays
            Y2 = np.hstack([Y, y_aug])
            X = []
            for i in range(X2.shape[0]):
                X.append(X2[i])
            Y = Y2.tolist()
        
X = np.asarray(X)      
Y = np.asarray(Y)

np.save('../data/x_train_6_1920', X) #save
np.save('../data/y_train_6_1920', Y)

le = LabelEncoder() #create and fit label encoder
le = le.fit(Y.flatten())
y2 = le.transform(Y.flatten()) 

n_classes = y2.max()+1 #14

#scale data using a standard scalar (ie, substract mean and divide by std)
scalers = {}
for i in range(X.shape[1]):
    scalers[i] = StandardScaler()
    scalers[i].fit(X[:, i, :]) 

pickle.dump(scalers, open('../data/scaler_6_1920.pkl','wb')) #save
pickle.dump(le, open('../data/labelencoder_6_1920.pkl','wb'))
