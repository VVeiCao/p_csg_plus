import os, sys

import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
from conf.config import GlobalConfig

# change it to what you want
###############################################################################
root_dir = 'carla_data/clear-weather/data'
# self.train_towns = ['Town07', 'Town10']
towns = ['Town04'] # ['Town01', 'Town02', 'Town06', 'Town07', 'Town10'] #
types = ['tiny']
##############################################################################
data = []
for town in towns:
    for t in types:
        data.append(os.path.join(root_dir, town+'_'+t))

"""
This file aims to pre-process the rgb, segmentation and lidar input so that shorten the I/O time and accerate the training.
"""
ids = {'None':0, 'Buildings':1, 'Fences':2, 'Other':3, 'Pedestrians':4, 'Pole':5, 'RoadLines':6, 'Roads':7, 'Sidewalks':8, 'Vegetation':9, 'Vehicles':10, 'Walls':11, 'TrafficSigns':12, 'Sky':13, 'Ground': 14, 'Bridge': 15, 'RailTrack': 16, 'GuardRail':17, 'TrafficLight':18, 'Static':19, 'Dynamic':20, 'Water':21, 'Terrain':22}
id_obstacle = [4,10]
id_road = [3,6,7,27]
id_traffic_red_light = [23]
config = GlobalConfig()

def img_pre_processing(image, scale=1, crop=[256, 256], shift_x=0, shift_y=0):
    return torch.from_numpy(np.array(scale_and_crop_image(Image.open(image), scale=scale, crop=crop, shift_x=shift_x, shift_y=shift_y)))

def seg_pre_processing(seg, scale=1, crop=[256, 256], shift_x=0, shift_y=0):
    t = np.array(scale_and_crop_seg(Image.open(seg), scale=scale, crop=crop, shift_x=shift_x, shift_y=shift_y))
    return torch.from_numpy(t)

def scale_and_crop_seg(seg, scale=1, crop=[256, 256], shift_x=0, shift_y=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(seg.width // scale), int(seg.height // scale))
    seg_resized = seg.resize((width, height), Image.NEAREST)
    seg = np.asarray(seg_resized)
    start_x = height//2 - crop[0]//2 + shift_x
    start_y = width//2 - crop[1]//2 + shift_y
    cropped_seg = seg[start_x:start_x+crop[0], start_y:start_y+crop[1]]
    cropped_seg = cropped_seg
    return cropped_seg

def scale_and_crop_image(image, scale=1, crop=[256, 256], shift_x=0, shift_y=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop[0]//2 + shift_x
    start_y = width//2 - crop[1]//2 + shift_y
    cropped_image = image[start_x:start_x+crop[0], start_y:start_y+crop[1]]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image

def shrink_front_images(route_dir, filename):
    rgb_front_path = os.path.join(route_dir,"rgb_front",filename)
    rgb_front = img_pre_processing(rgb_front_path, scale=2.0, crop=[300, 400])
    im = Image.fromarray(rgb_front.permute(1, 2, 0).numpy())
    front_s_path = os.path.join(route_dir,"rgb_front_s")
    if not os.path.exists(front_s_path):
        os.makedirs(front_s_path)
    im.save(os.path.join(front_s_path,filename))

def stich_rgb_images(route_dir, filename):
    rgb_front_path = os.path.join(route_dir,"rgb_front",filename)
    rgb_left_path = os.path.join(route_dir,"rgb_left",filename)
    rgb_right_path = os.path.join(route_dir,"rgb_right",filename)

    rgb_front = img_pre_processing(rgb_front_path, scale=1.0, crop=[160, 300])
    rgb_left = img_pre_processing(rgb_left_path, scale=0.8, crop=[160, 234], shift_x=0, shift_y=-47)
    rgb_right = img_pre_processing(rgb_right_path, scale=0.8, crop=[160, 234], shift_x=0, shift_y=47)

    rgb_lfr = torch.cat((rgb_left, rgb_front, rgb_right), axis=2)
    im = Image.fromarray(rgb_lfr.permute(1, 2, 0).numpy())
    lfr_path = os.path.join(route_dir,"rgb_front_w")
    if not os.path.exists(lfr_path):
        os.makedirs(lfr_path)
    im.save(os.path.join(lfr_path,filename))

def stich_seg_images(route_dir, filename):
    seg_front_path = os.path.join(route_dir,"seg_front",filename)
    seg_left_path = os.path.join(route_dir,"seg_left",filename)
    seg_right_path = os.path.join(route_dir,"seg_right",filename)

    seg_front = seg_pre_processing(seg_front_path, scale=1.0, crop=[160, 300])
    seg_left = seg_pre_processing(seg_left_path, scale=0.8, crop=[160, 234], shift_x=0, shift_y=-47)
    seg_right = seg_pre_processing(seg_right_path, scale=0.8, crop=[160, 234], shift_x=0, shift_y=47)
    seg_lfr = torch.cat((seg_left, seg_front, seg_right), axis=1)
    im = Image.fromarray(seg_lfr.numpy())
    lfr_path = os.path.join(route_dir,"seg_front_w")
    if not os.path.exists(lfr_path):
        os.makedirs(lfr_path)
    im.save(os.path.join(lfr_path,filename))

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = config.pixels_per_meter
        hist_max_per_pixel = config.hist_max_per_pixel
        x_meters_max = config.x_meters_max
        y_meters_max = config.y_meters_max
        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.flip(np.transpose(features, (2, 1, 0)), 2).astype(np.uint8)
    return features

def lidar_preprocessing(route_dir, filename):
    lidar_path = os.path.join(route_dir,"lidar",filename)
    lidar_unprocessed = np.load(lidar_path)[...,:3] # lidar: XYZI
    lidar_unprocessed[:,1] *= -1
    lidar_processed = lidar_to_histogram_features(lidar_unprocessed, crop=256)
    # lidar_processed = lidar_processed
    path = os.path.join(route_dir,"lidar_p")
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path,filename), lidar_processed)

scale = 1.0
def process():
    for sub_root in tqdm(data, file=sys.stdout):
        print(os.getcwd())
        root_files = os.listdir(sub_root)
        routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
        for route in routes:
            route_dir = os.path.join(sub_root, route)
            num = len(os.listdir(route_dir+"/rgb_front/"))
            for i in range(num):
                filename = f"{str(i).zfill(4)}.png"
                lidar_filename = f"{str(i).zfill(4)}.npy"
                # shrink_front_images(route_dir, filename)
                stich_rgb_images(route_dir, filename)
                stich_seg_images(route_dir, filename)
                lidar_preprocessing(route_dir, lidar_filename)



process()
# route_dir = "carla_data/weather-mixed/data/Town10_tiny/routes_town10_tiny_w-1_05_06_23_00_32"
# num = len(os.listdir(route_dir+"/rgb_front/"))
# for i in range(num):
#     filename = f"{str(i).zfill(4)}.png"
#     lidar_filename = f"{str(i).zfill(4)}.npy"
#     stich_rgb_images(route_dir, filename)