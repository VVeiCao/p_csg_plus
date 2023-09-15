
import copy
from PIL import Image


from torchvision import transforms
import math

import argparse
import json
import os
import pdb
from matplotlib import image
from tqdm import tqdm

import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from conf.attack_config import GlobalConfig
from models.p_csg import P_CSG
from dataset.data_new import CARLA_Data
from utils.utils import calculate_penalty, log_name



def plot_img_from_normalized_img(img_array, is_normalized=True):
    img_to_be_plotted = copy.deepcopy(img_array)
    assert len(img_array.shape) == 3
    if img_to_be_plotted.shape[0] == 3:
        img_to_be_plotted = img_to_be_plotted.transpose(1, 2, 0)
    if is_normalized:
        for idx, (m, v) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            img_to_be_plotted[:, :, idx] = (img_to_be_plotted[:, :, idx] * v) + m
    return img_to_be_plotted



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


class AttackModel(nn.Module):
    """
    Class to create an adversarial example.
    forward method returns the prediction result of the perturbated image.
    """
    def __init__(self,config,args):
        super(AttackModel, self).__init__()
        self.image_dot = ImageDot()
        # self.base_model = models.resnet50(pretrained=True).eval()
        model = P_CSG(config).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.logdir,'best_model.pth')))
        self.base_model = model.eval()
        self.device = args.device

        self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, fronts, lidars, target_point, gt_measurements):
        fronts = self.image_dot(fronts) # normlized
        inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        inv_fronts = inv_normalize(fronts)
        inv_fronts = inv_fronts * 255

        return self.base_model(inv_fronts, lidars, target_point, gt_measurements) # PCSG's input is also normlized


def create_tqdm_bar(iterable,desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc, leave=False)


def train(model, train_loader,tb_logger, config, args, num_of_runs):
    epochs = args.epochs
    name = args.name

    training_loss = 0
    lr = 0.008

    extent = (2.450842, 1.064162, 0.755373) # (1 / 2 * （lenght, width, height）)
    r = np.pi / 2
    c, s = np.cos(r), np.sin(r)
    r_matrix = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation_matrix_pi2 = torch.from_numpy(r_matrix).to(args.device, dtype=torch.float32)

    for epoch in range(epochs):
        if (epoch + 1) % 5 == 0:
            lr *= 0.5
        model.zero_grad()
        training_loop = create_tqdm_bar(train_loader, desc=f"Training Epoch [{epoch}/{epochs}]")
        for train_iteration, data in training_loop:
            # create batch and move to GPU
            fronts = data['front'][0].to(args.device, dtype=torch.float32)
            lidars = data['lidar'][0].to(args.device, dtype=torch.float32)

            front_seg = data['seg_front'][0].to(args.device, dtype=torch.float32)
            topdown_seg = data['seg_topdown'][0].to(args.device, dtype=torch.float32)
            topdown_seg_ext = data['seg_topdown_ext'][0].to(args.device, dtype=torch.float32)
            # driving labels
            command = data['command'].to(args.device)
            gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
            gt_steer = data['steer'].to(args.device, dtype=torch.float32)
            gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
            gt_theta = torch.stack(data['theta'], dim=1).to(args.device, dtype=torch.float32)
            gt_brake = data['brake'].to(args.device, dtype=torch.float32)
            gt_measurements = torch.stack([gt_velocity,gt_steer,gt_throttle,gt_brake], 1)
            gt_traffic_light = torch.stack((torch.isin(data['light'], torch.tensor([23,24])), data['light'] == 25, ~torch.isin(data['light'], torch.tensor([23,24,25]))), dim=1).to(args.device, dtype=torch.float32)
            gt_stop_sign = torch.stack((data['stop_sign']==1, data['stop_sign']==0), dim=1).to(args.device, dtype=torch.float32)
            gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)

            # target point
            target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

            # traffic light informations
            light = data["light"].to(args.device, dtype=torch.float32)
            stop_sign = data["stop_sign"].to(args.device, dtype=torch.bool)

            

            fronts = fronts / 255

            # imagejpg = Image.fromarray(fronts[0].cpu().numpy().transpose(1,2,0).astype('uint8'))
            # imagejpg.save('attack/p_csg/image_output/rgb_ori.png')    
            
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            fronts = norm(fronts)

            # model_dict = torch.load('attack/p_csg/logs/loss_attack_pcsg_1/run_2/470_ckpt.tar')
            # model = ImageDot()
            # model.load_state_dict(model_dict)
            # rgb = model(fronts)
            # inv_normalize = transforms.Normalize(
            #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            #         std=[1/0.229, 1/0.224, 1/0.225]
            #     )
            # rgb = inv_normalize(rgb)
            # rgb = rgb * 255

            # imagejpg = Image.fromarray(perturbed_rgb[0].cpu().numpy().transpose(1,2,0).astype('uint8'))
            # imagejpg.save('image_output/rgb_dot0.jpg')
            ret = model(fronts, lidars, target_point, gt_measurements)
            losses = model.base_model.losses(
                ret,
                gt_waypoints, 
                gt_traffic_light, 
                gt_stop_sign,
                front_seg, 
                topdown_seg,
                topdown_seg_ext
            )
            wp_loss, wp_loss_zero, front_seg_loss, td_seg_loss, td_seg_ext_loss, tl_loss, ss_loss, alignment_loss = \
                losses['wp_loss'], losses['wp_loss_zero'], losses['front_seg_loss'], losses['td_seg_loss'], losses['td_seg_ext_loss'], losses['tl_loss'], losses['ss_loss'], losses['alignment_loss'] 

            w1 = args.td_rec_loss_w
            w1_1 = args.td_ext_rec_loss_w
            w2 = args.front_rec_loss_w
            w3 = args.tl_indicator_loss_w
            w4 = args.ss_indicator_loss_w
            w5 = args.feature_alignment_w

            auxiliary_loss = w1 * td_seg_loss + w1_1 * td_seg_ext_loss + w2 * front_seg_loss + w3 * tl_loss + w4 * ss_loss + w5 * alignment_loss

            # red light penalty calculation
            if (torch.sum(light == 23) != 0 or torch.sum(light == 24) != 0):
                red_light = torch.logical_or((light == 23), (light == 24))
                red_light_penalty = calculate_penalty(ret['pred_wp'], light, rotation_matrix_pi2, data, extent,is_pred_waypoints=True).mean()
                red_light_penalty_gt = calculate_penalty(gt_waypoints, light, rotation_matrix_pi2, data, extent)
                red_light_violation = red_light_penalty_gt > 0

                loss_pos_obey_rules = torch.cat((wp_loss[~red_light], wp_loss[red_light][~red_light_violation], wp_loss_zero[red_light][red_light_violation]))
                loss = args.lambda1 * red_light_penalty + loss_pos_obey_rules.mean() if loss_pos_obey_rules.nelement() != 0 else args.lambda1 * red_light_penalty
            else:
                red_light_penalty = torch.tensor(0.0)
                loss = wp_loss.mean()	

            # speed penalty calculation
            desired_speed = torch.linalg.norm(ret['pred_wp'][:, 0, :] - ret['pred_wp'][:, 1, :], dim=-1) * 2.0
            d_theta = gt_theta[:, -1] - gt_theta[:, 0]
            speed_penalty = torch.abs(torch.sin(d_theta)) * (torch.clamp(desired_speed, min=args.speed_lb)- args.speed_lb)
            speed_penalty = speed_penalty.mean()
            loss = loss + args.lambda2 * speed_penalty

            # stop sign penalty calculation
            desired_speed = torch.linalg.norm(ret['pred_wp'][:, 0, :] - ret['pred_wp'][:, 1, :], dim=-1) * 2.0
            if stop_sign.any():
                stop_sign_penalty = stop_sign.float() * torch.clamp(desired_speed - args.speed_lb_ss, min=0.0)
                stop_sign_penalty = stop_sign_penalty.mean()
                loss += args.lambda3 * stop_sign_penalty

            loss += auxiliary_loss

            model_loss = - loss
            model_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0,norm_type=2.0)
            for param in model.parameters():
                if param.requires_grad == True:
                    print(param)
                    param.data = torch.clamp(param.data - param.grad.data * lr, min=0.0, max=1.0)
                    if torch.isnan(param).any(): raise ValueError('Value is nan')
            if epoch == epochs - 1:
                training_loss += model_loss.item()

            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(model_loss.item()))
            tb_logger.add_scalar(f'{name}/train_loss', model_loss.item(), epoch * len(train_loader) + train_iteration)

        if epoch != 0 and epoch % 10 ==0:
            save_dir = os.path.join('attack','p_csg','logs',name, f'run_{num_of_runs}',f'{epoch}_ckpt.tar')
            torch.save(model.image_dot.state_dict(),save_dir)

            # load ckpt
            # model_dict = torch.load('logs/stop_sign_attack/run_29/ckpt.tar')
            # model = ImageDot()
            # model.load_state_dict(model_dict)
            # test = model(test.unsqueeze(0))

        


# args
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=1000, help='Number of total epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--logdir', type=str, default='model_ckpts/ckpt_final', help='Directory to best model.')
parser.add_argument('--save_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--name', type=str, default='loss_attack_pcsg_final')


parser.add_argument('--id', type=str, default='p-csg', help='Unique experiment identifier.')
parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
parser.add_argument('--save_all_models', type=bool, default=False, help='if save all models during the training')
parser.add_argument('--td_rec_loss_w', type=float, default=0.5, help='weight of topdown reconstraction loss')
parser.add_argument('--td_ext_rec_loss_w', type=float, default=0.5, help='weight of topdown extra reconstraction loss')
parser.add_argument('--front_rec_loss_w', type=float, default=0.5, help='weight of left-front-right reconstraction loss')
parser.add_argument('--tl_indicator_loss_w', type=float, default=0.5, help='weight of traffic light indicator loss')
parser.add_argument('--ss_indicator_loss_w', type=float, default=0.5, help='weight of stop sign indicator loss')
parser.add_argument('--feature_alignment_w', type=float, default=0.1, help='the weight of the feature alignment loss')
parser.add_argument('--lambda1', type= float, default=0.5, help='weight of red light penalty')
parser.add_argument('--lambda2', type= float, default=0.05, help='weight of speed penalty')
parser.add_argument('--lambda3', type= float, default=0.5, help='weight of stop sign penalty')
parser.add_argument('--speed_lb', type= float, default=2.5, help='speed lower bound to avoid punishment')
parser.add_argument('--speed_lb_ss', type= float, default=1.0, help='speed lower bound to avoid stop sign punishment')
parser.add_argument('--save_program_files', type= bool, default=True, help='flag for saving program files in experiment path before each training')

args = parser.parse_args()

# Config
config = GlobalConfig()

# model
model = AttackModel(config,args)
# Data
train_set = CARLA_Data(root=config.train_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

path = os.path.join('attack/p_csg/logs',args.name)
num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
path = os.path.join(path,f'run_{num_of_runs+1}')
tb_logger = SummaryWriter(path)

train(model,dataloader_train,tb_logger,config,args, num_of_runs+1 )