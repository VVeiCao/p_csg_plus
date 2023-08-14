import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import copy
from .utils import seg_one_hot_key, seg_pre_processing, transform_2d_points, lidar_pre_processing, rgb_specific_seg_pre_processing, scale_and_crop_image, img_pre_processing_org

class CARLA_Data(Dataset):
    def __init__(self, root, config):
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        self.input_resolution = config.input_resolution
        self.td_scale = config.td_scale
        self.root = root
        self.data = {
            'lidar': [],
            'front': [],
            'seg_front': [],
            'seg_topdown': [],
            'seg_topdown_ext': [],
            'x': [],
            'y': [],
            'x_command': [],
            'y_command': [],
            'theta': [],
            'steer': [],
            'throttle': [],
            'brake': [],
            'command': [],
            'velocity': [],
            'light': [],
            'stop_sign': [],
            'value': [],
            'direction_1': [],
            'direction_2': []
        }
        self.data_temp = copy.deepcopy(self.data)
        self._load_data()

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.data['front'])
    
    def __getitem__(self, index):
        train_data = {
            'front': [],
            'lidar': [],
            'seg_front': [],
            'seg_topdown': [],
            'seg_topdown_ext': []
        }
        seq_x = self.data['x'][index]
        seq_y = self.data['y'][index]
        seq_theta = self.data['theta'][index]

        for i in range(self.seq_len):
            train_data['front'].append(img_pre_processing_org(np.array(Image.open(self.data['front'][index][i]))))
            train_data['seg_front'].append(seg_one_hot_key(np.array(Image.open(self.data['seg_front'][index][i]))))
            train_data['seg_topdown'].append(seg_pre_processing(self.data['seg_topdown'][index][i], self.td_scale, self.input_resolution, shift_x=-128))
            train_data['seg_topdown_ext'].append(rgb_specific_seg_pre_processing(self.data['seg_topdown'][index][i], self.td_scale, self.input_resolution, shift_x=-128))
            train_data['lidar'].append(lidar_pre_processing(self.data['lidar'][index][i]))

        ego_x = self.data['x'][index][i]
        ego_y = self.data['y'][index][i]
        ego_theta = self.data['theta'][index][i]

        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

        train_data['waypoints'] = waypoints

        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
        ])
        local_command_point = np.array([self.data['x_command'][index]-ego_x, self.data['y_command'][index]-ego_y])
        local_command_point = R.T.dot(local_command_point)
        train_data['target_point'] = tuple(local_command_point)
        keys = ['steer', 'throttle', 'brake', 'command', 'velocity', 'theta', 'light', 'value', 'direction_1', 'direction_2', 'stop_sign']
        for k in keys:
            train_data[k] = self.data[k][index]

        train_data["xy"] = torch.from_numpy(np.array([ego_y, -ego_x])) # crrent coordinate x, y => world coordinate y, -z

        r1, t1_x, t1_y = np.pi / 2 - ego_theta, -ego_x, -ego_y
        c, s = np.cos(r1), np.sin(r1)
        r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])
        train_data['rotation_matrix'] = torch.from_numpy(r1_to_world)

        return train_data

    def _load_data(self):
        """
            load data from prepared files or create files and load them
        """
        for sub_root in tqdm(self.root, file=sys.stdout):
            # TODO: change name
            preload_file = os.path.join(sub_root, 'rgbw_lidar' + str(self.seq_len) + '_' + str(self.pred_len) + '_pre_load.npy')
            # dump to npy if no preload
            if not os.path.exists(preload_file):
            # if True:
                preload_data = copy.deepcopy(self.data_temp)
                # list sub-directories in root
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
                    for seq in range(num_seq):
                        seq_data = {
                            'front': [],
                            'lidar': [],
                            'seg_front': [],
                            'seg_topdown': [],
                            'seg_topdown_ext': [],
                            'x': [],
                            'y': [],
                            'theta': []
                        }
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq*self.seq_len+1+i).zfill(4)}.png"
                            seq_data['front'].append(os.path.join(route_dir,"rgb_front_w",filename))

                            # segmentations
                            seq_data['seg_front'].append(os.path.join(route_dir,"seg_front_w",filename))
                            seq_data['seg_topdown'].append(os.path.join(route_dir,"topdown",filename))
                            seq_data['seg_topdown_ext'].append(os.path.join(route_dir,"topdown",filename))
                            # point cloud
                            seq_data['lidar'].append(route_dir + f"/lidar_p/{str(seq*self.seq_len+1+i).zfill(4)}.npy")

                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            seq_data['x'].append(data['gps_x'])
                            seq_data['y'].append(data['gps_y'])
                            if np.isnan(data['theta']):
                                seq_data['theta'].append(0.)
                            else:
                                seq_data['theta'].append(data['theta'])

                        with open(route_dir + f"/light/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                            data_light = json.load(read_file)

                        with open(route_dir + f"/stop/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                            data_stop_sign = json.load(read_file)
                        
                        preload_data['stop_sign'].append(data_stop_sign['stop'])

                        
                        if data_light['light_crossing'] == 0 or data_light['light_crossing'] == -1 or data_light['crossing'] == 1:
                            # if ego car has already crossed the stopline, the traffic light can be detected 
                            preload_data['light'].append(0)
                            preload_data['value'].append(0.0)
                            preload_data['direction_1'].append(0)
                            preload_data['direction_2'].append(0)
                        else:
                            preload_data['light'].append(data_light['light_crossing'])
                            preload_data['value'].append(data_light['value'])
                            preload_data['direction_1'].append(data_light['direction_1'])
                            preload_data['direction_2'].append(data_light['direction_2'])

                        with open(route_dir + f"/measurements/{str(seq*self.seq_len).zfill(4)}.json", "r") as read_file_lastframe:
                            data_lastframe = json.load(read_file_lastframe)

                        with open(route_dir + f"/measurements/{str(seq*self.seq_len+1).zfill(4)}.json", "r") as read_file_lastframe:
                            data = json.load(read_file_lastframe)

                        # get control value of final frame in sequence
                        preload_data['x_command'].append(data['x_command'])
                        preload_data['y_command'].append(data['y_command'])
                        preload_data['steer'].append(data_lastframe['steer'])
                        preload_data['throttle'].append(data_lastframe['throttle'])
                        preload_data['brake'].append(data['brake'])
                        preload_data['command'].append(data['command'])
                        preload_data['velocity'].append(data['speed'])

                        # read files sequentially (future frames)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            # point cloud
                            seq_data['lidar'].append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            seq_data['x'].append(data['gps_x'])
                            seq_data['y'].append(data['gps_y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                seq_data['theta'].append(0)
                            else:
                                seq_data['theta'].append(data['theta'])
                        keys = {'front', 'lidar', 'seg_front', 'seg_topdown', 'x', 'y', 'theta'}
                        for key in keys:
                            preload_data[key].append(seq_data[key])

                np.save(preload_file, preload_data)
            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            for key in self.data.keys():
                self.data[key] += preload_dict.item()[key]

            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

        