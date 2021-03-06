import json
import time
import os
from typing import Tuple

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fs_s3fs import S3FS
from fs.copy import copy_dir, copy_file

from tensorflow.keras.metrics import CategoricalAccuracy, MeanIoU

from eolearn.core import EOPatch, FeatureType, LoadTask, SaveTask, OverwritePermission
from eoflow.models.segmentation_unets import ResUnetA

from eoflow.models.losses import TanimotoDistanceLoss

from eoflow.models.metrics import MCCMetric

from .utils import BaseConfig, prepare_filesystem, set_sh_config


@dataclass
class PredictionConfig(BaseConfig):
    eopatches_folder: str
    feature_extent: Tuple[FeatureType, str]
    feature_boundary: Tuple[FeatureType, str]
    feature_distance: Tuple[FeatureType, str]
    model_path: str
    model_name: str
    model_version: str
    temp_model_path: str
    height: int
    width: int
    n_channels: int
    n_classes: int
    metadata_path: str
    batch_size: int
    normalise: str
    feature_bands: Tuple[FeatureType, str] = (FeatureType.DATA, 'BANDS')
    reference_extent: Tuple[FeatureType, str] = (FeatureType.MASK_TIMELESS, 'EXTENT')
    reference_boundary: Tuple[FeatureType, str] = (FeatureType.MASK_TIMELESS, 'BOUNDARY')
    reference_distance: Tuple[FeatureType, str] = (FeatureType.DATA_TIMELESS, 'DISTANCE')


def binary_one_hot_encoder(array: np.ndarray) -> np.ndarray:
    """ One hot encode the label array along the last dimension """
    return np.concatenate([1 - array, array], axis=-1)


def crop_array(array: np.ndarray, buffer: int = 5) -> np.ndarray:
    """ Crop height and width of a 4D array given a buffer size. Array has shape B x H x W x C """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return array[:, 9*buffer+1:-buffer*9:, buffer+3:-(buffer+4):, :]


def pad_array(array: np.ndarray, buffer: int = 5) -> np.ndarray:
    """ Pad height and width dimensions of a 4D array with a given buffer. Height and with are in 2nd and 3rd dim """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return np.pad(array, [(0, 0), (buffer*9+1, buffer*9), (buffer+3, buffer+4), (0, 0)], mode='edge')


def get_tanimoto_loss(from_logits: bool = False) -> TanimotoDistanceLoss:
    return TanimotoDistanceLoss(from_logits=from_logits)


def get_accuracy_metric(name: str = 'accuracy') -> CategoricalAccuracy:
    return CategoricalAccuracy(name=name)


def get_iou_metric(n_classes: int, name: str = 'iou') -> MeanIoU:
    return MeanIoU(num_classes=n_classes, name=name)


def get_mcc_metric(n_classes: int, threshold: float = .5) -> MCCMetric:
    mcc_metric = MCCMetric(default_n_classes=n_classes, default_threshold=threshold)
    mcc_metric.init_from_config({'n_classes': n_classes})
    return mcc_metric


def prediction_fn(eop: EOPatch, n_classes: int,
                  normalization_factors: pd.DataFrame,
                  normalise: str,
                  model: ResUnetA, model_name: str,
                  extent_feature: Tuple[FeatureType, str],
                  boundary_feature: Tuple[FeatureType, str],
                  distance_feature: Tuple[FeatureType, str],
                  suffix: str,
                  batch_size: int,
                  bands_feature: Tuple[FeatureType, str],
                  reference_extent: Tuple[FeatureType, str],
                  reference_boundary: Tuple[FeatureType, str],
                  reference_distance: Tuple[FeatureType, str]) -> EOPatch:
    """ Perform prediction for all timestamps in an EOPatch given a model and normalization factors """
    assert normalise in ['to_meanstd', 'to_medianstd']
      
    extent_pred, boundary_pred, distance_pred = [], [], []
    metrics = []

    padded = pad_array(eop[bands_feature])
    padded = padded[:,:,:,[1,2,3,7]]*10000
    
    print(f'padded_shape: {padded.shape}')
    
    tanimoto_loss = get_tanimoto_loss()
    accuracy_metric = get_accuracy_metric()
    iou_metric = get_iou_metric(n_classes=n_classes)
    mcc_metric = get_mcc_metric(n_classes=n_classes)

    for timestamp, bands in zip(eop.timestamp, padded):

        month = timestamp.month

        norm_factors_month = normalization_factors[normalization_factors['month'] == month].iloc[0]

        dn_mean = np.array([norm_factors_month.norm_meanstd_mean_b0,
                            norm_factors_month.norm_meanstd_mean_b1,
                            norm_factors_month.norm_meanstd_mean_b2,
                            norm_factors_month.norm_meanstd_mean_b3])

        dn_median = np.array([norm_factors_month.norm_meanstd_median_b0,
                              norm_factors_month.norm_meanstd_median_b1,
                              norm_factors_month.norm_meanstd_median_b2,
                              norm_factors_month.norm_meanstd_median_b3])

        dn_std = np.array([norm_factors_month.norm_meanstd_std_b0,
                           norm_factors_month.norm_meanstd_std_b1,
                           norm_factors_month.norm_meanstd_std_b2,
                           norm_factors_month.norm_meanstd_std_b3])

        avg_stat = dn_mean if normalise == 'to_meanstd' else dn_median
        data = (bands - avg_stat) / dn_std
        print(f'np.min(bands): {np.min(bands)}')
        print(f'np.max(bands): {np.max(bands)}')
        print(f'np.min(data): {np.min(data)}')
        print(f'np.max(data): {np.max(data)}')
        print(f'avg_stat: {avg_stat}')
        print(f'dn_std: {dn_std}')
        
        
        start_time = time.time()
        extent, boundary, distance = model.net.predict(data[np.newaxis, ...], batch_size=batch_size)
        print(f'Time for prediction: {time.time() - start_time} seconds')
        
        print(f'extent_predicted_shape: {extent.shape}')
        extent = crop_array(extent)
        boundary = crop_array(boundary)
        distance = crop_array(distance)
        print(f'extent_PredictedAndCropped_shape: {extent.shape}')
        extent_pred.append(extent)
        boundary_pred.append(boundary)
        distance_pred.append(distance)

        tmp = {}
        for mask_name, gt, pred in [('extent', eop[reference_extent], extent),
                                    ('boundary', eop[reference_boundary], boundary),
                                    ('distance', eop[reference_distance], distance)]:
            tmp[f'{mask_name}_loss'] = tanimoto_loss(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
            tmp[f'{mask_name}_acc'] = accuracy_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
            tmp[f'{mask_name}_iou'] = iou_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()
            tmp[f'{mask_name}_mcc'] = mcc_metric(binary_one_hot_encoder(gt[np.newaxis, ...]), pred).numpy()[1]

            accuracy_metric.reset_states()
            iou_metric.reset_states()
            mcc_metric.reset_states()

        metrics.append(tmp)

    extent_pred = np.concatenate(extent_pred, axis=0)
    boundary_pred = np.concatenate(boundary_pred, axis=0)
    distance_pred = np.concatenate(distance_pred, axis=0)

    ### [...,[1]] gets one of the two values and add "1" dimension => [H,W,1]
    ### Without [1], it would be [H,W]
    eop[extent_feature] = extent_pred[..., [1]]
    eop[boundary_feature] = boundary_pred[..., [1]]
    eop[distance_feature] = distance_pred[..., [1]]

    eop.meta_info[f'metrics_{suffix}'] = metrics
    eop.meta_info[f'model_{suffix}'] = model_name

    return eop


def load_metadata(filesystem: S3FS, config: PredictionConfig) -> pd.DataFrame:
    """ Load DataFrame with info about normalisation factors """
    metadata_dir = os.path.dirname(config.metadata_path)
#     if not filesystem.exists(metadata_dir):
#         filesystem.makedirs(metadata_dir)

    df = pd.read_csv(open(f'{config.metadata_path}'))

    normalization_factors = df.groupby(pd.to_datetime(df.timestamp).dt.to_period("M")).max()

    normalization_factors['month'] = pd.to_datetime(normalization_factors.timestamp).dt.month

    return normalization_factors


def load_model(filesystem: S3FS, config: PredictionConfig) -> ResUnetA:
    """ Copy the model locally if not existing and load it """
#     if not os.path.exists(f'{config.temp_model_path}/{config.model_name}'):
#         if not os.path.exists(f'{config.model_path}/{config.model_name}/checkpoints/'):
#             os.makedirs(f'{config.model_path}/{config.model_name}/checkpoints/')
#         copy_dir(filesystem, f'{config.model_path}/{config.model_name}/checkpoints/',
#                  f'{config.temp_model_path}/{config.model_name}', 'checkpoints')
#         copy_file(filesystem, f'{config.model_path}/{config.model_name}/model_cfg.json',
#                   f'{config.temp_model_path}/{config.model_name}', 'model_cfg.json')

    input_shape = dict(features=[None, config.height, config.width, config.n_channels])

    with open(f'{config.temp_model_path}/{config.model_name}/model_cfg.json', 'r') as jfile:
        model_cfg = json.load(jfile)

    # initialise model from config, build, compile and load trained weights
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()
    model.net.load_weights(f'{config.temp_model_path}/{config.model_name}/checkpoints/model.ckpt')

    return model


def run_prediction_on_eopatch(eopatch_name: str, config: PredictionConfig,
                              model: ResUnetA = None, normalization_factors: pd.DataFrame = None) -> dict:
    """ Run prediction workflow on one eopatch. Model and dataframe can be provided to avoid loading them every time """
#     sh_config = set_sh_config(config)

#     filesystem = prepare_filesystem(config)

    filesystem = None
    
    if normalization_factors is None:
        normalization_factors = load_metadata(filesystem, config)

    if model is None:
        model = load_model(filesystem, config)

    load_task = LoadTask(path=f'{config.eopatches_folder}',
                         features=[config.feature_bands,
                                   config.reference_distance, config.reference_extent, config.reference_boundary,
                                   FeatureType.TIMESTAMP,
                                   FeatureType.META_INFO,
                                   FeatureType.BBOX])

    save_task = SaveTask(path=f'{config.eopatches_folder}',
                         features=[config.feature_extent, config.feature_boundary, config.feature_distance,
                                   FeatureType.META_INFO],
                         overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

#     try:

    eop = load_task.execute(eopatch_folder=eopatch_name)

    eop = prediction_fn(eop,
                        normalization_factors=normalization_factors,
                        normalise=config.normalise,
                        model=model, model_name=config.model_name,
                        extent_feature=config.feature_extent,
                        boundary_feature=config.feature_boundary,
                        distance_feature=config.feature_distance,
                        suffix=config.model_version,
                        batch_size=config.batch_size,
                        n_classes=config.n_classes,
                        bands_feature=config.feature_bands,
                        reference_boundary=config.reference_boundary,
                        reference_distance=config.reference_distance,
                        reference_extent=config.reference_extent)

    _ = save_task.execute(eop, eopatch_folder=eopatch_name)

    del eop

    return dict(name=eopatch_name, status='Success')

#     except Exception as exc:
#         return dict(name=eopatch_name, status=exc)
