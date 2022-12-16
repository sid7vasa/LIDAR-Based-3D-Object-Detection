# Includes partially modified code from available helper functions. 

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir("/home/kishore/workspace/LIDAR-Based-3D-Object-Detection/")
import sys
sys.path.append("/home/kishore/workspace/LIDAR-Based-3D-Object-Detection/")
import gc
import numpy as np
import pandas as pd
from functools import partial
import glob 
from multiprocessing import Pool

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List
import yaml

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

from matplotlib.axes import Axes
from matplotlib import animation, rc
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot, init_notebook_mode
import plotly.figure_factory as ff

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm
from tqdm import tqdm, tqdm_notebook, notebook

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from data.dataset_object import LyftDataset

def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
    
    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """
    
    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)
    
    tm = np.eye(4, dtype=np.float32)
    translation = shape/2 + offset/voxel_size
    
    tm = tm * np.array(np.hstack((1/voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm  

def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3,4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]

def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p

def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):

    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
    points_voxel_coords = np.int0(points_voxel_coords)
    
    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))
    
    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)
        
    # Note X and Y are flipped:
    bev[coord[:,1], coord[:,0], coord[:,2]] = count
    
    return bev

def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev/max_intensity).clip(0,1)

def move_boxes_to_car_space(boxes, ego_pose):
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse
    
    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        box.rotate(rotation)
        
def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor

def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord

        class_color = classes.index(box.name) + 1
        
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)

def visualize_lidar_of_sample(sample_token, level5data, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)

def prepare_training_data_for_scene(level5data, first_sample_token, output_folder, bev_shape, voxel_size, z_offset, box_scale, classes, *args, **kwargs):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    sample_token = first_sample_token
    
    while sample_token:
        
        sample = level5data.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])


        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                            inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue
        
        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        
        boxes = level5data.get_boxes(sample_lidar_token)

        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev*255).astype(np.uint8)
        target_im = target[:,:,0] # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
        
        sample_token = sample["next"]

def bev_data_creation(config):
    DATA_PATH = config['DATA_PATH']
    json_path = config['json_path']
    lyft_dataset = LyftDataset(DATA_PATH, json_path=json_path)
    ARTIFACTS_FOLDER = config["artifacts"]
    level5data = LyftDataset(data_path='.', json_path=json_path, verbose=True)
    os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

    # Extract scene data from the dataset and create a dataframe
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    records = [(level5data.get('sample', record['first_sample_token'])['timestamp'], record) for record in level5data.scene]
    entries = []

    # Iterate over the timestamp storted data and append into a list
    for start_time, record in sorted(records):
        start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000
        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]
        entries.append((host, name, date, token, first_sample_token))
                
    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
    host_count_df = df.groupby("host")['scene_token'].count()

    # Split data into train and validation
    validation_hosts = ["host-a007", "host-a008", "host-a009"]
    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]

    # Create a sample data to work generate BEV representation.
    sample_token = train_df.first_sample_token.values[0]
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = level5data.get("sample_data", sample_lidar_token)
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

    ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
    calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    # Homogeneous transformation matrix from car frame to world frame.
    global_from_car = transform_matrix(ego_pose['translation'],
                                    Quaternion(ego_pose['rotation']), inverse=False)
    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                        inverse=False)
    
    # There is an error in one of the lidar data. so ignore the specific file.
    spec_lidar_path_1 = '/home/kishore/workspace/lidar_data/data/3d-object-detection-for-autonomous-vehicles/train_lidar/host-a011_lidar1_1233090652702363606.bin'
    if str(lidar_filepath) != spec_lidar_path_1:
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
 

    # The lidar pointcloud is defined in the sensor's reference frame.
    # We want it in the car's reference frame, so we transform each point
    lidar_pointcloud.transform(car_from_sensor)
    
    # Some hyperparameters we'll need to define for the system
    voxel_size = config["voxel_size"]
    z_offset = config["z_offset"]
    bev_shape = config["bev_shape"]

    # We scale down each box so they are more separated when projected into our coarse voxel space.
    box_scale = config["box_scale"]

    bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)

    # So that the values in the voxels range from 0,1 we set a maximum intensity.
    bev = normalize_voxel_intensities(bev)
   
    boxes = level5data.get_boxes(sample_lidar_token)

    target_im = np.zeros(bev.shape[:3], dtype=np.uint8)

    move_boxes_to_car_space(boxes, ego_pose)
    scale_boxes(boxes, box_scale)
    draw_boxes(target_im, voxel_size, boxes, classes, z_offset=z_offset)
    # visualize_lidar_of_sample(sample_token, level5data)
    del bev, lidar_pointcloud, boxes
    # "bev" stands for birds eye view
    train_data_folder = config["train_data_folder"]
    validation_data_folder = config["validation_data_folder"]
    NUM_WORKERS = 1
    for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
        print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
        first_samples = df.first_sample_token.values

        os.makedirs(data_folder, exist_ok=True)
        
        process_func = partial(prepare_training_data_for_scene,
                            level5data, output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale, classes=classes)

        pool = Pool(NUM_WORKERS)
        for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
            pass
        pool.close()
        del pool

if __name__ == "__main__":
    #
    # fixed args
    #    
    default_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    with open("./parameters/train.yaml", "r") as f:
        parameters = yaml.safe_load(f)

    # Some hyperparameters we'll need to define for the system
    voxel_size = (0.4,0.4,1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)
    box_scale = 0.8

    config ={}
    config["DATA_PATH"] = parameters["DATA_PATH"]
    config["json_path"] = config["DATA_PATH"]  + "train_data"
    config["artifacts"] = "./res"
    config["train_data_folder"] = os.path.join(config["artifacts"], "bev_train_data")
    config["validation_data_folder"] = os.path.join(config["artifacts"], "./bev_validation_data")
    config["voxel_size"] = voxel_size
    config["z_offset"] = z_offset
    config["bev_shape"] = "./res"

    bev_data_creation(config)