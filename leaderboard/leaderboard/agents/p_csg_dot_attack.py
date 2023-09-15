import os
import json
import datetime
import pathlib
import pdb
import time
import cv2
import carla
import torch.nn as nn
from collections import deque
import copy
from copy import deepcopy
import torch
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.models.p_csg import P_CSG
from team_code.conf.config import GlobalConfig
from team_code.dataset.utils import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points, \
    img_pre_processing_org
from planner import RoutePlanner

import math
from matplotlib import cm

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'PCSGAgent'


class PCSGAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file,route):
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.unified = None

        self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(),
                             'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

        self.config = GlobalConfig()

        self.stuck_detector = 0
        self.forced_move = 0
        self.net = P_CSG(self.config)
        self.emergency_stop = False
        self.net.to("cuda")
        self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))  # TODO: change back to the best model
        self.net.cuda()
        self.net.eval()
        self.net.inference = True
        self.route = route
        if self.config.save_frames:
            now = datetime.datetime.now()
            string = 'Town05_long_'  # pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb_front').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'seg_front').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'seg_topdown').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 1.3, 'y': 0.0, 'z': 2.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'id': 'lidar'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0
        lidar = input_data['lidar'][1][:, :3]

        result = {
            'rgb': rgb,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'rgb_rear': rgb_rear,
            'lidar': lidar,
            'gps': gps,
            'speed': speed,
            'compass': compass,
        }

        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        if self.step < self.config.seq_len:
            rgb_front = torch.from_numpy(
                scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=1.0, crop=[160, 300])).unsqueeze(0)
            rgb_left = torch.from_numpy(
                scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=0.8, crop=[160, 234], shift_x=0,
                                     shift_y=-47)).unsqueeze(0)
            rgb_right = torch.from_numpy(
                scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=0.8, crop=[160, 234], shift_x=0,
                                     shift_y=47)).unsqueeze(0)
            rgb = torch.cat((rgb_left, rgb_front, rgb_right), dim=-1)
            self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))

            self.input_buffer['lidar'].append(tick_data['lidar'])
            self.input_buffer['gps'].append(tick_data['gps'])
            self.input_buffer['thetas'].append(tick_data['compass'])

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            self.global_throttle = 0.0
            self.global_steer = 0.0
            self.global_brake = 0.0

            return control

        gt_velocity = tick_data['speed']

        # if self.old_light == 0 and self.accumulation >= 20.0:
        # 	gt_velocity = 2.5
        # elif self.old_light == 1 and self.accumulation >= 10.0:
        # 	gt_velocity = 2.5
        # if self.accumulation >= 58.0:
        # 	gt_velocity = 4

        gt_velocity = torch.FloatTensor([gt_velocity]).to('cuda', dtype=torch.float32)

        command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)
        gt_throttle = torch.FloatTensor([self.global_throttle]).to('cuda', dtype=torch.float32)
        gt_steer = torch.FloatTensor([self.global_steer]).to('cuda', dtype=torch.float32)
        gt_brake = torch.FloatTensor([self.global_brake]).to('cuda', dtype=torch.float32)
        gt_theta = torch.FloatTensor([tick_data['compass']]).to('cuda', dtype=torch.float32)
        # gt_measurements = torch.cat([gt_velocity, gt_steer, gt_throttle, gt_brake]).view(4, -1)
        gt_measurements = torch.stack([gt_velocity, gt_steer, gt_throttle, gt_brake], 1)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                     torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        encoding = []
        rgb_front = torch.from_numpy(
            scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=1.0, crop=[160, 300])).unsqueeze(0)
        rgb_left = torch.from_numpy(
            scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=0.8, crop=[160, 234], shift_x=0,
                                 shift_y=-47)).unsqueeze(0)
        rgb_right = torch.from_numpy(
            scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=0.8, crop=[160, 234], shift_x=0,
                                 shift_y=47)).unsqueeze(0)
        rgb = torch.cat((rgb_left, rgb_front, rgb_right), dim=-1)

        rgb = rgb / 255
        from torchvision import transforms
        composed = transforms.Compose(
                [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        rgb = composed(rgb.to(torch.float))
        model_dict = torch.load('attack/p_csg/dot_ckpt/dot_attack_ckpt.tar')
        model = ImageDot()
        model.load_state_dict(model_dict)
        rgb = model(rgb)
        inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        rgb = inv_normalize(rgb)
        rgb = rgb * 255



        self.input_buffer['rgb'].popleft()
        self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))

        self.input_buffer['lidar'].popleft()
        self.input_buffer['lidar'].append(tick_data['lidar'])
        self.input_buffer['gps'].popleft()
        self.input_buffer['gps'].append(tick_data['gps'])
        self.input_buffer['thetas'].popleft()
        self.input_buffer['thetas'].append(tick_data['compass'])

        # transform the lidar point clouds to local coordinate frame
        ego_theta = self.input_buffer['thetas'][-1]
        ego_x, ego_y = self.input_buffer['gps'][-1]

        if self.step % 2 == 0:  # since safety check uses lidar
            # safety check
            safety_box = deepcopy(tick_data['lidar'])
            safety_box[:, 1] *= -1  # inverts x, y
            # z-axis
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            self.emergency_stop = (len(safety_box) > self.config.safety_box_n)  # Checks if there is object before the agent

        # Only predict every second step because we only get a LiDAR every other frame.
        if self.step % 2 == 0 or self.step <= 4:
            for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
                curr_theta = self.input_buffer['thetas'][i]
                curr_x, curr_y = self.input_buffer['gps'][i]
                lidar_point_cloud[:, 1] *= -1  # inverts x, y
                lidar_transformed = transform_2d_points(lidar_point_cloud,
                                                        np.pi / 2 - curr_theta, -curr_x, -curr_y, np.pi / 2 - ego_theta,
                                                        -ego_x, -ego_y)
                lidar_transformed = torch.from_numpy(
                    lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
                self.lidar_processed = list()
                self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))
            unified = self.input_buffer['rgb']

            self.unified = copy.deepcopy(unified)
            self.ret = self.net(unified[0], self.lidar_processed[0], target_point, gt_measurements)

        light_prob = torch.nn.functional.softmax(self.ret[1].cpu().squeeze(0), dim=0)
        traffic_light = torch.argmax(light_prob, dim=0)  # (red/yellow, green, no light)
        # unblock
        is_stuck = False
        # divide by 2 because we process every second frame
        # 1100 = 55 seconds * 20 Frames per second, we move for 1.5 second = 30 frames to unblock
        if self.stuck_detector > self.config.stuck_threshold and traffic_light != 0 and self.forced_move < self.config.creep_duration:
            print("Detected agent being stuck. Move for frame: ", self.forced_move)
            is_stuck = True
            self.forced_move += 1
        elif self.stuck_detector > self.config.block_threshold and self.forced_move < self.config.creep_duration:
            print("Detected agent being blocked. Move to prevent time out!")
            is_stuck = True
            self.forced_move += 2
            self.emergency_stop = False



        steer, throttle, brake, metadata = self.net.control_pid(self.ret[0], torch.FloatTensor([gt_velocity]),
                                                                is_stuck)
        self.pid_metadata = metadata
        self.global_steer = steer
        self.global_throttle = throttle
        self.global_brake = brake
        if gt_velocity < 0.1:  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        elif gt_velocity > 0.1 and is_stuck is False:
            self.stuck_detector = 0
            self.forced_move = 0

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # print('steer: ', steer, 'throttle: ', throttle, 'brake: ', brake)

        if self.emergency_stop and is_stuck:  # We only use the saftey box when unblocking
            print("Detected object directly in front of the vehicle. Stopping. Step:", self.step)
            control.steer = float(steer)
            control.throttle = float(0.0)
            control.brake = float(True)

            output = self.unified[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.uint8(output)).save(pathlib.Path('/home/ubuntu/projects/p_csg_final/frames') / 'rgb' / ('%04d.png' % self.step))

        if self.config.save_frames and self.step % self.config.save_frequency == 0:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.config.save_frequency

        output = self.unified[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        filename = f"{self.route}_{frame:04d}.png"
        Image.fromarray(np.uint8(output)).save(self.save_path / 'rgb' / filename)

        # Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(
        #     self.save_path / 'lidar_0' / ('%04d.png' % frame))
        # Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(
        #     self.save_path / 'lidar_1' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net



class ImageDot(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha =  nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),requires_grad=True)
        self.beta =  nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),requires_grad=True)
        self.center = nn.Parameter(torch.tensor([
            [0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
            [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
            [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            requires_grad=True)

    def forward(self, x):
        _, _, height, width = x.shape
        blended = x
        for idx in range(self.center.shape[0]):
            scale = torch.tensor([(height-1), (width-1)])
            mask = self._create_circle_mask(height, width,
                                            self.center[idx] * scale , self.beta[idx] * 2.0, self.radius[idx] * 25.0)
            normalized_color = self._normalize_color(self.color[idx],
                                                     self.means, self.stds)
            blended = self._create_blended_img(blended, mask, normalized_color, self.alpha[idx] * 0.3)
        return blended

    def _normalize_color(self, color, means, stds):
        return list(map(lambda x, m, s: (x - m) / s, color, means, stds))

    def _create_circle_mask(self, height, width, center, beta, radius):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        d = ((hv - center[0]) ** 2 + (wv - center[1]) ** 2) / radius ** 2
        return torch.exp(- d ** beta + 1e-10)

    def _create_blended_img(self, base, mask, color, alpha):
        
        mask = mask.to(base.device)
        alpha_tile = alpha * mask.expand(3, mask.shape[0], mask.shape[1])
        color_tile = torch.zeros_like(base)
        for c in range(3):
            color_tile[:, c, :, :] = color[c]
        return (1. - alpha_tile) * base + alpha_tile * color_tile

