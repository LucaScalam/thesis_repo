from abc import abstractmethod
import os
import boto3
import fs
from fs_s3fs import S3FS

import pyproj
from shapely.ops import transform

from shapely.geometry import Polygon,MultiPolygon
from datetime import datetime, timedelta
import dateutil
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tqdm.notebook import tqdm
from typing import List, Union, Tuple, Optional

from skimage.morphology import binary_dilation, disk
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

from sentinelhub import CRS, BBox

from eolearn.core import FeatureType, EOPatch, EOTask, EOWorkflow, LinearWorkflow, LoadTask, SaveTask, \
    OverwritePermission, EOExecutor, SaveToDisk
from eolearn.geometry import VectorToRaster


class DB2Vector(EOTask):
    """
    Reads vectors to EOPatch from a local postgre db.
    """

    def __init__(self, vector_output_feature):
        self.database=""
        self.user = ""
        self.password = ""
        self.host = ""
        self.port = ""
        self.crs = CRS.WGS84.pyproj_crs()
        self.out_vector = next(self._parse_features(vector_output_feature)())

    def execute(self, eopatch):
        utm_crs = eopatch.bbox.crs.pyproj_crs()
        project = pyproj.Transformer.from_proj(utm_crs, self.crs)
        
        query_bbox = transform(project.transform, eopatch.bbox.geometry)

        spatial_query = 'select * from gsaa where ST_Intersects(ST_GeomFromText((%s), (%s)), geom);'
        parameters = (query_bbox.wkt, self.crs.to_epsg())

        with psycopg2.connect(database=self.database, user=self.user, password=self.password, 
                              host=self.host, port=self.port) as con:
            df = gpd.GeoDataFrame.from_postgis(spatial_query, con, geom_col='geom', params=parameters)
            df = df.to_crs(utm_crs).rename(columns={'geom':'geometry'}).set_geometry('geometry')
        
        eopatch[self.out_vector] = df

        return eopatch
    
    
class Extent2Boundary(EOTask):
    """
    Adds boundary mask from extent mask using binary dilation
    """
    def __init__(self, extent_feature, boundary_feature, structure=None):
        self.extent_feature = next(self._parse_features(extent_feature)())
        self.boundary_feature = next(self._parse_features(boundary_feature)())
        self.structure = structure
        
    def execute(self, eopatch):
        extent_mask = eopatch[self.extent_feature].squeeze(axis=-1)
        boundary_mask = binary_dilation(extent_mask, selem=self.structure) - extent_mask
        eopatch[self.boundary_feature] = boundary_mask[..., np.newaxis]
        
        return eopatch
    
    
class Extent2Distance(EOTask):
    """
    Adds boundary mask from extent mask using binary dilation
    """
    def __init__(self, extent_feature, distance_feature, normalize=True):
        self.extent_feature = next(self._parse_features(extent_feature)())
        self.distance_feature = next(self._parse_features(distance_feature)())
        self.normalize = normalize
        
    def execute(self, eopatch):
        extent_mask = eopatch[self.extent_feature].squeeze(axis=-1)
        
        distance = distance_transform_edt(extent_mask)
        
        if not self.normalize:
        
            eopatch[self.distance_feature] = distance[..., np.newaxis]

            return eopatch
        
        conn_comp = label(extent_mask, background=0)
        unique_comp = np.unique(conn_comp)
        normalised = np.zeros(distance.shape, dtype=np.float32)
        
        for uc in unique_comp:
            if uc != 0:
                conn_comp_mask = conn_comp==uc
                normalised[conn_comp_mask] += distance[conn_comp_mask]/np.max(distance[conn_comp_mask])
            
        eopatch[self.distance_feature] = normalised[..., np.newaxis]

        return eopatch
    
    
def get_workflow_vector_to_eopatch(load_task, save_task, vector, extent_feature, raster_shape, boundary_feature, distance_feature):
    """ Function to create a workflow that will load extent, boundary and distance on the eopatch. """
    
    ### get extent mask from vector
    vecRas = VectorToRaster(
        vector_input = vector,
        raster_feature = extent_feature,
        values=1, raster_shape=raster_shape,
        no_data_value=0, buffer=-10)

    ### get boundary mask from extent mask
    ras2bound = Extent2Boundary(extent_feature, 
                                boundary_feature,
                                structure=disk(2))
    ### get distance from extent mask
    ras2dist = Extent2Distance(extent_feature, 
                               distance_feature, 
                               normalize=True)

    ### workflow
    workflow = LinearWorkflow(load_task, vecRas, ras2bound, ras2dist, save_task)
    
    return workflow