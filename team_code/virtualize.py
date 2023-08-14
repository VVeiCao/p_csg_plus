import argparse
import json
from locale import normalize
import os
import pdb
# from typing_extensions import get_overloads
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from conf.config import GlobalConfig
from models.p_csg import P_CSG
from dataset.data_new import CARLA_Data

from PIL import Image

import matplotlib.pyplot as plt

ids = {'None':0, 'Buildings':1, 'Fences':2, 'Other':3, 'Pedestrians':4, 'Pole':5, 'RoadLines':6, 'Roads':7, 'Sidewalks':8, 'Vegetation':9, 'Vehicles':10, 'Walls':11, 'TrafficSigns':12, 'Sky':13, 'Ground': 14, 'Bridge': 15, 'RailTrack': 16, 'GuardRail':17, 'TrafficLight':18, 'Static':19, 'Dynamic':20, 'Water':21, 'Terrain':22}
id_obstacle = [1,2,4,5,8,9,10,11,12,14,15,16,17,18,19,20,21,22]
id_road = [3,6,7,27]
id_traffic_light = [18]

red = np.array([149, 53, 83],dtype=np.uint8) # obstacle
blue = np.array([112, 137, 215], dtype = np.uint8) # road
white = np.array([255,255,255],dtype=np.uint8) # void
orange = np.array([228, 147, 24], dtype=np.uint8) # roadlines
green = np.array([170, 255, 0], dtype=np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='/home/zhk/project/vae/output/', help='Directory to log data')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='batch size')
parser.add_argument('--virtualize_rate', type=int, default=7, help='batch size')
args = parser.parse_args()

# Config
config = GlobalConfig()
group = '25_01_2023'
time = '18:21:20'
args.logdir = os.path.join(args.logdir, group, time)
# writer = SummaryWriter(log_dir=args.logdir)

use_cuda=True

def paint2(seg):
    (w,h) = seg.shape
    image = np.zeros([w,h,3], dtype=np.uint8)
    image[seg==0] = white
    return image

def paint(seg):
    (w,h) = seg.shape
    image = np.zeros([w,h,3], dtype=np.uint8)
    ## 0:obstacle
    ## 1:road
    ## 2:trafficLight
    ## 3:Others

    image[seg==0] = red
    image[seg==1] = blue
    #image[seg[3]==1] = red

    return image

def virtualize(model, device, dataloader):
    model.eval()
    num = 0
    for data in tqdm(dataloader):
        # create batch and move to GPU
        fronts = data['front'][0].to(args.device, dtype=torch.float32)
        lidars = data['lidar'][0].to(args.device, dtype=torch.float32)

        # driving labels
        command = data['command'].to(args.device)
        gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
        gt_steer = data['steer'].to(args.device, dtype=torch.float32)
        gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
        gt_brake = data['brake'].to(args.device, dtype=torch.float32)
        gt_theta = torch.stack(data['theta'], dim=1).to(args.device, dtype=torch.float32)
        gt_measurements = torch.stack([gt_velocity,gt_steer,gt_throttle,gt_brake],1)

        # target point
        target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

        ret = model(fronts, lidars, target_point, gt_measurements)
        # print(nn.functional.cross_entropy(ret['topdown_rec'], topdown_seg[0]))
        front_rec = nn.functional.softmax(ret['front_rec'], dim=1).cpu().detach().numpy()
        front_rec = np.argmax(front_rec[0], axis=0)
        front_rec_img = paint(front_rec)

        front_gt = data['seg_front']
        front_gt = np.argmax(front_gt[0][0], axis=0)
        front_gt_img = paint(front_gt)

        front_cat = np.concatenate((front_rec_img, front_gt_img), axis=0)


        topdown_rec = nn.functional.softmax(ret['topdown_rec'], dim=1).cpu().detach().numpy()
        topdown_rec = np.argmax(topdown_rec[0], axis=0)
        topdown_rec_img = paint(topdown_rec)

        topdown_gt = data['seg_topdown']
        topdown_gt = np.argmax(topdown_gt[0][0], axis=0)
        topdown_gt_img = paint(topdown_gt)

        topdown_cat = np.concatenate((topdown_rec_img, topdown_gt_img), axis=1)

        topdown_ext_rec = nn.functional.softmax(ret['topdown_rec_ext'], dim=1).cpu().detach().numpy()
        topdown_ext_rec = np.argmax(topdown_ext_rec[0], axis=0)
        topdown_ext_rec_img = paint2(topdown_ext_rec)

        topdown_ext_gt = data['seg_topdown_ext']
        topdown_ext_gt = np.argmax(topdown_ext_gt[0][0], axis=0)
        topdown_ext_gt_img = paint2(topdown_ext_gt)

        topdown_ext_cat = np.concatenate((topdown_ext_rec_img, topdown_ext_gt_img), axis=1)

        if num % args.virtualize_rate == 0:
            Image.fromarray(topdown_cat).save("topdown_img/img" + str(num) + ".jpg")
            Image.fromarray(front_cat).save("front_img/img" + str(num) + ".jpg")
            Image.fromarray(topdown_ext_cat).save("topdown_ext_img/img" + str(num) + ".jpg")
            # Image.fromarray(lidar_onroad_combined_data).save("rec_lidar/onroad" + str(num) + ".jpg")
            # Image.fromarray(lidar_overroad_combined_data).save("rec_lidar/overroad" + str(num) + ".jpg")
        num+=1

        
# Config
config = GlobalConfig()

# Data
val_set = CARLA_Data(root=config.val_data, config=config)

dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = P_CSG(config).to(args.device)

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print ('Total trainable parameters: ', params)

# Load checkpoint
model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))

# Log args
# with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
# 	json.dump(args.__dict__, f, indent=2)

virtualize(model, args.device, dataloader_val)