import numpy as np 
import torch
from PIL import Image

ids = {'None':0, 'Buildings':1, 'Fences':2, 'Other':3, 'Pedestrians':4, 'Pole':5, 'RoadLines':6, 'Roads':7, 'Sidewalks':8, 'Vegetation':9, 'Vehicles':10, 'Walls':11, 'TrafficSigns':12, 'Sky':13, 'Ground': 14, 'Bridge': 15, 'RailTrack': 16, 'GuardRail':17, 'TrafficLight':18, 'Static':19, 'Dynamic':20, 'Water':21, 'Terrain':22}
id_obstacle = [4,10,17]
id_road = [3,6,7,26,27]
id_road_line = [6]
id_traffic_red_light = [23,24]
id_traffic_green_light = [25]
id_stop_sign = [26, 27]

def img_pre_processing(image, scale=1, crop=256, shift_x=0, shift_y=0):
    return torch.from_numpy(np.array(scale_and_crop_image(Image.open(image), scale=scale, crop=crop, shift_x=shift_x, shift_y=shift_y)))

def img_pre_processing_org(image):
    return torch.from_numpy(np.transpose(image,(2,0,1)))
    
def seg_pre_processing(seg, scale=1, crop=256, shift_x=0, shift_y=0):
    t = np.array(scale_and_crop_seg(Image.open(seg), scale=scale, crop=crop, shift_x=shift_x, shift_y=shift_y))
    obstacle = np.isin(t, id_obstacle).astype(float)
    road = np.isin(t, id_road).astype(float)
    others = 1.0-np.isin(t, id_obstacle + id_road).astype(float)
    t = np.stack([obstacle, road, others], axis=0)

    ## 0:obstacle
    ## 1:road
    ## 2:Others

    return torch.from_numpy(t)

def rgb_specific_seg_pre_processing(seg, scale=1, crop=256, shift_x=0, shift_y=0):
    t = np.array(scale_and_crop_seg(Image.open(seg), scale=scale, crop=crop, shift_x=shift_x, shift_y=shift_y))
    roadline = np.isin(t, id_road_line).astype(float)
    # red_light = np.isin(t, id_traffic_red_light).astype(float)
    # green_light = np.isin(t, id_traffic_green_light).astype(float)
    # stop_sign = np.isin(t, id_stop_sign).astype(float)
    others = 1.0-np.isin(t, id_road_line).astype(float)
    t = np.stack([roadline, others], axis=0)

    ## 0: roadline
    ## 4: others
    return torch.from_numpy(t)

def lidar_pre_processing(lidar):
    t = torch.from_numpy((np.load(lidar).astype(np.float32)))
    t /= torch.max(t)
    return t
    
def seg_one_hot_key(seg):
    """
    Args:
        seg: segmentation with labels
    Return:
        one hot key representation of the segmentation
    """
    obstacle = np.isin(seg, id_obstacle).astype(float)
    road = np.isin(seg, id_road).astype(float)
    traffic_red_light = np.isin(seg, id_traffic_red_light).astype(float)
    others = 1.0-np.isin(seg, id_obstacle + id_road).astype(float)
    t = np.stack([obstacle, road, others], axis=0)
    return torch.from_numpy(t)

def scale_and_crop_seg(seg, scale=1, crop=[256,256], shift_x=0, shift_y=0):
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
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.flip(np.transpose(features, (2, 1, 0)), 2).astype(np.float32)
    return features