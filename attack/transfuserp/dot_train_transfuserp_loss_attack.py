import argparse
import json
from torchvision import transforms
import PIL
import copy
import os
from tqdm import tqdm
import numpy as np
import torch
import datetime
import pathlib
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from config import GlobalConfig
# from v3.models.p_csg import P_CSG
from model import LidarCenterNet
# from late_fusion.model import LateFusion
import torch.nn.functional as F
from data import CARLA_Data, lidar_bev_cam_correspondences
# from vae.dataset.data_new import CARLA_Data,CARLA_Data_Stop,CARLA_Data_Light
import random
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



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
        args.logdir = os.path.join(args.logdir, args.id)
        # self.base_model = models.resnet50(pretrained=True).eval()

        # /home/cw/Desktop/physical-world-attack/transfuserp/transfuser+/model_ckpt/transfuser2/model_26.pth
        parallel = bool(args.parallel_training)

        shared_dict = None


        rank       = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:{}'.format(local_rank))

        torch.cuda.set_device(device)

        torch.backends.cudnn.benchmark = True # Wen want the highest performance

        # Configure config
        config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
        config.use_target_point_image = bool(args.use_target_point_image)
        config.n_layer = args.n_layer
        config.use_point_pillars = bool(args.use_point_pillars)
        config.backbone = args.backbone
        if(bool(args.no_bev_loss)):
            index_bev = config.detailed_losses.index("loss_bev")
            config.detailed_losses_weights[index_bev] = 0.0       
            
            
        model = LidarCenterNet(config,args.device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
        # model.load_state_dict(torch.load("model_ckpt/transfuser2/model_26.pth"))
        model.load_state_dict(torch.load(self.load_file))

        self.base_model = model.eval()
        self.device = args.device

        self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, fronts, lidar, ego_waypoint = None, target_point= None,
                        target_point_image= None,
                        ego_vel = None, bev= None,
                        label= None, save_path= None,
                        depth= None, semantic= None, num_points = None):
        fronts = self.image_dot(fronts) # normlized
        inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        inv_fronts = inv_normalize(fronts)

        return self.base_model.forward(inv_fronts, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                        target_point_image=target_point_image,
                        ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                        label=label, save_path= None,
                        depth=depth, semantic=semantic, num_points=num_points) 


def create_tqdm_bar(iterable,desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc, leave=False)


def train(model, train_loader,tb_logger, config, args, num_of_runs):
    epochs = args.epochs
    name = args.name
    device = args.device

    training_loss = 0
    lr = 0.008

    detailed_losses = config.detailed_losses
    detailed_losses_weights = config.detailed_losses_weights
    detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(detailed_losses)}
    
    for epoch in range(epochs):
        if (epoch + 1) % 1 == 0:
            lr *= 0.5
        model.zero_grad()
        training_loop = create_tqdm_bar(train_loader, desc=f"Training Epoch [{epoch}/{epochs}]")
        for train_iteration, data in training_loop:
            loss = torch.tensor(0.0).to(device, dtype=torch.float32)
            
            rgb = data['rgb'].to(device, dtype=torch.float32) # [20, 3, 160, 704])
            composed = transforms.Compose(
                    [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            rgb = composed(rgb)
            
            if config.multitask:
                depth = data['depth'].to(device, dtype=torch.float32)
                semantic = data['semantic'].squeeze(1).to(device, dtype=torch.long)
            else:
                depth = None
                semantic = None

            bev = data['bev'].to(device, dtype=torch.long)

            if (config.use_point_pillars == True):
                lidar = data['lidar_raw'].to(device, dtype=torch.float32)
                num_points = data['num_points'].to(device, dtype=torch.int32)
            else:
                lidar = data['lidar'].to(device, dtype=torch.float32)
                num_points = None

            label = data['label'].to(device, dtype=torch.float32)
            ego_waypoint = data['ego_waypoint'].to(device, dtype=torch.float32)

            target_point = data['target_point'].to(device, dtype=torch.float32)
            target_point_image = data['target_point_image'].to(device, dtype=torch.float32)

            ego_vel = data['speed'].to(device, dtype=torch.float32)
           
            losses = model(fronts = rgb,lidar = lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                        target_point_image=target_point_image,
                        ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                        label=label, save_path= None,
                        depth=depth, semantic=semantic, num_points=num_points)
        
            for key, value in losses.items():
                loss += detailed_weights[key] * value

            model_loss = - loss#-loss_3 #+loss_0
            model_loss.backward(retain_graph=True)

            nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0,norm_type=2.0)

            for param in model.parameters():
                if param.requires_grad == True:
                    param.data = torch.clamp(param.data - param.grad.data * lr, min=0.0, max=1.0)
            if epoch == epochs - 1:
                training_loss += model_loss.item()

            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(loss.item()))
            tb_logger.add_scalar(f'{name}/train_loss', model_loss.item(), epoch * len(train_loader) + train_iteration)


        if epoch != 0 and epoch % 5 ==0:
            save_dir = os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_ckpt.tar')
            torch.save(model.image_dot.state_dict(),save_dir)



# args
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=1000, help='Number of total epochs')
parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate.')
parser.add_argument('--logdir', type=str, default='output/', help='Directory to log data to.')
parser.add_argument('--save_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--name', type=str, default='loss_attack_transfuserp')


parser.add_argument('--id', type=str, default='transfuserp_attack', help='Unique experiment identifier.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
parser.add_argument('--load_file', type=str, default="/home/ubuntu/projects/P_CSG/attack/transfuser/model_ckpt", help='ckpt to load.')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
parser.add_argument('--schedule', type=int, default=1,
                    help='Whether to train with a learning rate schedule. 1 = True')
parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                    help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                    help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
parser.add_argument('--backbone', type=str, default='transFuser',
                    help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
parser.add_argument('--image_architecture', type=str, default='regnety_032',
                    help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                    help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
parser.add_argument('--use_velocity', type=int, default=0,
                    help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
parser.add_argument('--wp_only', type=int, default=0,
                    help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
parser.add_argument('--use_target_point_image', type=int, default=1,
                    help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
parser.add_argument('--use_point_pillars', type=int, default=0,
                    help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
parser.add_argument('--parallel_training', type=int, default=0,
                    help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                        ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
parser.add_argument('--val_every', type=int, default=2, help='At which epoch frequency to validate.')
parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train normally, 1: set training weight for BEV to 0')
parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')

parser.add_argument('--root_dir', type=str, default=r'/home/data2/data/data/', help='Root directory of your training data')
parser.add_argument('--setting', type=str, default='05_only', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
args = parser.parse_args()

# Config
config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)

# #
# import os

# cpu_num = 32 # 这里设置成你想运行的CPU个数
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)

# import cv2
# cv2.setNumThreads(32)

# model
model = AttackModel(config,args)
# Data
# train_set = CARLA_Data(root=config.train_data, config=config)
train_set = CARLA_Data(root=config.train_data, config=config, shared_dict=None)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)



path = os.path.join('logs',args.name)
num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
path = os.path.join(path,f'run_{num_of_runs+1}')
tb_logger = SummaryWriter(path)

train(model,dataloader_train,tb_logger,config,args, num_of_runs+1 )






