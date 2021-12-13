from scipy import stats
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from enum import Enum
from tqdm import tqdm 
from contextlib import contextmanager  

import geopandas as gpd
from shapely.geometry import Polygon

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio import plot
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.crs

from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam

from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor, SaveTask 
from eolearn.io import S2L1CWCSInput, ExportToTiff
from eolearn.io.local_io import *
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask, AddMultiCloudMaskTask
from eolearn.mask.utilities import resize_images
from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask, InterpolationTask, ValueFilloutTask, \
    HaralickTask, AddMaxMinNDVISlopeIndicesTask, AddMaxMinTemporalIndicesTask, AddSpatioTemporalFeaturesTask, \
    HOGTask, MaxNDVICompositing, LocalBinaryPatternTask

# use context manager so DatasetReader and MemoryFile get cleaned up automatically
@contextmanager
def resample_raster(raster, size):
    """ Resamplig raster to size 'size'. This works when raster is 10m, 20m or 60m resolution. """
    t = raster.transform
    
    if (raster.height>size):
        scale = 1
    else:
        scale = size//raster.height

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = raster.height * scale
    width = raster.width * scale

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read( # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(raster.count, height, width, ),
            resampling=Resampling.bilinear,
        )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return     

class ImportFromJP2(BaseLocalIo):
   
    def __init__(self, feature, folder=None, *, timestamp_size=None, bad_data = [], **kwargs):
        
        super().__init__(feature, folder=folder, **kwargs)
        self.bad_data = bad_data
        self.timestamp_size = timestamp_size
        self.no_data_value  = np.nan


    @staticmethod
    def _get_reading_window(width, height, data_bbox, eopatch_bbox):
        """ Calculates a window in pixel coordinates for which data will be read from an image
        """
        if eopatch_bbox.crs is not data_bbox.crs:
            eopatch_bbox = eopatch_bbox.transform(data_bbox.crs)

        # The following will be in the future moved to sentinelhub-py
        data_ul_x, data_lr_y = data_bbox.lower_left
        data_lr_x, data_ul_y = data_bbox.upper_right

        res_x = abs(data_ul_x - data_lr_x) / width
        res_y = abs(data_ul_y - data_lr_y) / height

        ul_x, lr_y = eopatch_bbox.lower_left
        lr_x, ul_y = eopatch_bbox.upper_right

        # If these coordinates wouldn't be rounded here, rasterio.io.DatasetReader.read would round
        # them in the same way
        top = round((data_ul_y - ul_y) / res_y)
        left = round((ul_x - data_ul_x) / res_x)
        bottom = round((data_ul_y - lr_y) / res_y)
        right = round((lr_x - data_ul_x) / res_x)

        return (top, bottom), (left, right)
    
    def execute(self, eopatch=None, *, fn_exp='**/*B*.jp2'):

        feature_type, feature_name = next(self.feature())
        if eopatch is None:
            eopatch = EOPatch()

            
        #get jp2 files matching filename expression
        jp2_files = [[*f.stem.split('_'),f] for f in self.folder.glob(fn_exp)]
        # p2_files = [[*f.stem.split('_'),f] for f in self.folder.glob(fn_exp) if [*f.stem.split('_'),f][0] isin tiles_interest]
        jp2_df = pd.DataFrame(jp2_files)
        jp2_df.columns = ['tile','date','band','filename']
        jp2_df = jp2_df[(jp2_df.tile == 'T20HNH') | (jp2_df.tile == 'T20HPH')]
        
        band_order = {
                        'B01':0,
                        'B02':1, #Blue
                        'B03':2, #Green
                        'B04':3, #Red
                        'B05':4,
                        'B06':5,
                        'B07':6,
                        'B08':7, #NIR
                        'B8A':8,
                        'B09':9,
                        'B10':10,
                        'B11':11,
                        'B12':12
                    }

        jp2_df['band_order'] = jp2_df.band.map(band_order)

        jp2_df.sort_values(by=['date','tile','band_order'],inplace=True)
        
        dates = jp2_df.date.unique()
        tiles = jp2_df.tile.unique()
        bands = list(band_order.keys())
        print(f'tiles available: {tiles}')
        dates_list = dates.tolist() #list of possible dates
        date_data_list = list() #list to hold data for each date
        #iterate through all files and add data
        i = 0
        for date in dates:
#             print(self.bad_data)
            if date in self.bad_data:
                dates_list.remove(date)
                print(f'--> bad_data: {date}')
                continue
         
            print(f'--------------------- date: {i}, {date}')
            i = i+1
            bands_data_list = list() #list to hold data for each band
            for band in bands:
                tile_data_list = list()
                for tile in tiles:
                    filename_series = jp2_df[(jp2_df.date==date)
                                     &(jp2_df.band==band)
                                     &(jp2_df.tile==tile)].filename
                    if len(filename_series) > 0:
                        filename=filename_series.values[0]
                    else:
                        continue
                    
                    ### The commented block is for tiles with different crs, so it includes a projection.
                    ### However, there is a problem with merged_tile_data when projecting, because we get
                    ### narray with differents shapes. This needs a review in the future.
        
                    with rasterio.open(filename) as source:
#                         print(source.crs)
#                         print(rasterio.crs.CRS.from_epsg(32720))
#                         if source.crs != rasterio.crs.CRS.from_epsg(32720) : 
#                             print('reprojection in')
#                           ### Possible lines of code for reprojection before disjoint_bounds() function
#                             crs1 = rasterio.crs.CRS.from_epsg(32720)
#                             with rasterio.open(filename) as src:
#                                 transform, width, height = calculate_default_transform(
#                                     src.crs, crs1, src.width, src.height, *src.bounds)
#                                 kwargs = src.meta.copy()
#                                 kwargs.update({
#                                     'crs': crs1,
#                                     'transform': transform,
#                                     'width': width,
#                                     'height': height
#                                 })

#                                 with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
#                                     for i in range(1, src.count + 1):
#                                         reproject(
#                                             source=rasterio.band(src, i),
#                                             destination=rasterio.band(dst, i),
#                                             src_transform=src.transform,
#                                             src_crs=src.crs,
#                                             dst_transform=transform,
#                                             dst_crs=crs1,
#                                             resampling=Resampling.nearest)
#                                 source = rasterio.open('/tmp/RGB.byte.wgs84.tif') 
#                                 print(f'source shape{source.shape}')

                        with resample_raster(source, size = 10980) as resampled:
                            ### for 10m resolution, source.shape will be (10980,10980)
#                             print(f'resampled.shape: {resampled.shape}')
                            data_bbox = BBox(resampled.bounds, CRS(resampled.crs.to_epsg()))
                            
                            #If bounds are disjoint, change tile
                            if rasterio.coords.disjoint_bounds(rasterio.coords.BoundingBox(*data_bbox), 
                                                               rasterio.coords.BoundingBox(*eopatch.bbox)):
                                continue

                            ### Position of interest inside the EOPatch
                            reading_window = self._get_reading_window(
                                resampled.width, resampled.height, data_bbox, eopatch.bbox)

                            data = resampled.read(window=reading_window, boundless=True, 
                                                  fill_value=self.no_data_value)
#                             print(f'tile {tile}, data.shape {data.shape}')
                            if self.image_dtype is not None:
                                data = data.astype(self.image_dtype)
#                             print(data)    
                            tile_data_list.append(data)
#                             print(len(tile_data_list))

                ### Choose the values from the non-zero tile or average the two sets of non-zero value
                ### We are assuming no more than 2 tiles corresponding to each EOPatch
                if len(tile_data_list) > 1:
                    merged_tile_data = np.where((tile_data_list[0]!=0) & (tile_data_list[1]!=0),
                                            (tile_data_list[0]+tile_data_list[1])/2,
                                            np.where(tile_data_list[0]==0, tile_data_list[1], tile_data_list[0])
                                           )
                ### Extras
                elif len(tile_data_list) == 1:
                    merged_tile_data = tile_data_list[0]
                
                else:
                    continue
                bands_data_list.append(merged_tile_data.squeeze())
            
            ### Extras 
            if (len(bands_data_list)==0):
                #### delete date
                dates_list.remove(date)
                print(f'date: {date} removed')
                continue
            
            date_data_list.append(np.stack(bands_data_list,axis=-1))
        
        data = np.stack(date_data_list,axis=0)
#         eopatch[feature_type][feature_name] = data
        eopatch[feature_type][feature_name] = data/10000
        
        eopatch.timestamp = dates_list
        
        meta_info= {
            'service_type': 'wcs',
            'size_x': '10m',
            'size_y': '10m'
          }
        
        eopatch.meta_info = meta_info

        return eopatch

class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):        
        return np.logical_not(eopatch.mask['CLM'].astype(np.bool))
    
    
    
    
    
def import_jp2(feat_typ, path, bad_data):
    return ImportFromJP2(feature=[feat_typ], folder=Path(path), bad_data=bad_data)

def get_val_mask():
    return AddValidDataMaskTask(SentinelHubValidData(),'IS_DATA') # name of output mask