import torch 
from torch import gt, nn
from .pid_controller import PIDController
from .encoders.encoder import Encoder
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import pdb

class UnFlatten(nn.Module):
    def forward(self, input, size = 128):
        return input.view(input.size(0), size, 1, 1)

class P_CSG(nn.Module):
    '''
    Cross Semantic Generation Multi-sensor Fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.inference = False
        self.criterion = getattr(F, config.waypoint_loss_criterion)
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.encoder = Encoder(config)
        
        self.front_seg_channels = 3 # the number of segmetataion kinds; obstacle|road|others
        self.td_ext_seg_channels = 2 # roadline|others
        self.td_seg_channels = 3

        self.join = nn.Sequential(
                            nn.Linear(1156, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        )

        self.traffic_light_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

        self.stop_sign_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

        self.img_mu_s = nn.Linear(256, 128)
        self.img_lvar_s = nn.Linear(256, 128)

        self.img_mu_u = nn.Linear(256, 128)
        self.img_lvar_u = nn.Linear(256, 128)
        
        self.lidar_mu = nn.Linear(256, 128)
        self.lidar_lvar = nn.Linear(256, 128)
        
        self.front_seg_decoder =  nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,7), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,9), stride=2, output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(5,7), stride=(2,4), output_padding=(0,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(5,7), stride=(2,2), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(5,7), stride=(2,2), output_padding=(1,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(4, self.front_seg_channels, kernel_size=(5,7), stride=(2,2), output_padding=(1,1))
        )


        self.topdown_extra_seg_decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, self.td_ext_seg_channels, kernel_size=8, stride=2)
        )
        
        self.topdown_seg_decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, self.td_seg_channels, kernel_size=8, stride=2)
        )

        self.rgb_shared_features_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.rgb_unique_features_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.lidar_shared_features_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.lidar_unique_features_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.GRUCell(input_size=2, hidden_size=64)
        self.output = nn.Linear(64, 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def _img_bottleneck_s(self, h):
        mu, logvar = self.img_mu_s(h), self.img_lvar_s(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def _img_bottleneck_u(self, h):
        mu, logvar = self.img_mu_u(h), self.img_lvar_u(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar
    
    def _lidar_bottleneck(self, h):
        mu, logvar = self.lidar_mu(h), self.lidar_lvar(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar
    
    def _merge_gaussian(self, mu1, logvar1, mu2, logvar2):
        mu = 0.5 * (mu1 + mu2)
        logvar = torch.log(0.25 * logvar1.exp_() + 0.25 * logvar2.exp_()) # Simga = 1/4 * Sigma1 + 1/4 * Sigma2
        return mu, logvar

    def _load_weights(self):
        front_seg_channel_w = torch.Tensor(self.config.front_channel_weight).to(self.device, dtype=torch.float32)
        front_seg_channel_w /= front_seg_channel_w.sum()
        td_seg_channel_w = torch.Tensor(self.config.td_channel_weight).to(self.device, dtype=torch.float32)
        td_seg_channel_w /= td_seg_channel_w.sum()
        td_ext_seg_channel_w = torch.Tensor(self.config.td_ext_channel_weight).to(self.device, dtype=torch.float32)
        td_ext_seg_channel_w /= td_ext_seg_channel_w.sum()
        tl_w = torch.Tensor(self.config.tl_weight).to(self.device, dtype=torch.float32)
        tl_w /= tl_w.sum()
        ss_w = torch.Tensor(self.config.ss_weight).to(self.device, dtype=torch.float32)
        ss_w /= ss_w.sum()
        weights = {
            'front_seg_channel_w': front_seg_channel_w,
            'td_seg_channel_w':td_seg_channel_w,
            'td_ext_seg_channel_w':td_ext_seg_channel_w,
            'tl_w': tl_w,
            'ss_w': ss_w
        }
        return weights

    def wp_loss(self, pred_wp, gt_wp, reduction='mean'):
        return self.criterion(pred_wp, gt_wp, reduction=reduction)

    def wp_loss_zero(self, pred_wp, reduction='mean'):
        return self.criterion(pred_wp, torch.zeros_like(pred_wp), reduction=reduction)

    def seg_loss(self, rec_seg, gt_seg, mu, logvar, weight, reduction='mean'):
        ce_loss = F.cross_entropy(rec_seg.squeeze(), gt_seg.squeeze(), weight=weight, reduction=reduction)
        KLD_img = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return ce_loss + self.config.KL_weight * KLD_img

    def tl_loss(self, pred_traffic_light, gt_traffic_light, weight, reduction='mean'):
        return F.cross_entropy(pred_traffic_light, gt_traffic_light, weight=weight, reduction=reduction)
    
    def ss_loss(self, pred_stop_sign, gt_stop_sign, weight, reduction='mean'):
        return F.cross_entropy(pred_stop_sign, gt_stop_sign, weight=weight, reduction=reduction)

    def contrastive_alignment_loss(self, x, y, reduction='mean', bal=0.9):
        B, _ = x.shape
        x_square = torch.sum(torch.square(x), dim=-1)[:, None]
        y_square = torch.sum(torch.square(y), dim=-1)[None, :]
        xy = x @ y.mT
        euc_dis = torch.sqrt(x_square + y_square - 2. * xy)
        sim_mask = torch.eye(B).to(x)
        dissim_mask = 1. - sim_mask
        return bal * torch.sum(sim_mask * torch.square(euc_dis)) / B + (1-bal)*torch.sum(dissim_mask * torch.square(torch.clamp(1.-euc_dis, min=0))) / (B * (B-1))


    def kl_alignment_loss(self, mu1, logvar1, mu2, logvar2, mode=2):
        normal1 = Normal(mu1, logvar1.mul(0.5).exp_())
        normal2 = Normal(mu2, logvar2.mul(0.5).exp_())
        if mode == 0:
            loss = kl_divergence(normal1, normal2).mean()
        elif mode == 1:
            loss = kl_divergence(normal2, normal1).mean()
        else:
            loss = (kl_divergence(normal1, normal2).mean() + kl_divergence(normal2, normal1).mean()) / 2.0
        
        return loss

    # TODO: wasser stein distance calculation
    def wasser_stein_alignment_loss(self, mu1, logvar1, mu2, logvar2):
        pass

    def kl_contrastive_alignment_loss(self, mu1, logvar1, mu2, logvar2):
        B,_ = mu1.shape
        normal1 = Normal(mu1, logvar1.mul(0.5).exp_())
        normal2 = Normal(mu2, logvar2.mul(0.5).exp_())
        loss_sim = (kl_divergence(normal1, normal2).mean() + kl_divergence(normal2, normal1).mean()) / 2.0
        loss_dissim = 0.0
        for i in range(B-1):
            normal1 = Normal(torch.roll(mu1, shifts=i+1, dims=0), torch.roll(logvar1, shifts=i+1, dims=0).mul(0.5).exp_())
            normal2 = Normal(mu2, logvar2.mul(0.5).exp_())
            loss_dissim += torch.clip((kl_divergence(normal1, normal2).mean() + kl_divergence(normal2, normal1).mean()) / 2.0, max=0.01)
        loss_dissim /= (B-1)
        return loss_sim - loss_dissim 

    def losses(self, ret, gt_waypoints, gt_traffic_light, gt_stop_sign, front_seg, topdown_seg, topdown_seg_ext):
        weights = self._load_weights()
        wp_loss = self.wp_loss(ret['pred_wp'], gt_waypoints, reduction='none').mean(dim=[1,2])
        wp_loss_zero = self.wp_loss_zero(ret['pred_wp'], reduction='none').mean(dim=[1,2])
        front_seg_loss = self.seg_loss(ret['front_rec'], front_seg, ret['mu_front'], ret['logvar_front'], weight= weights['front_seg_channel_w'])
        td_seg_loss = self.seg_loss(ret['topdown_rec'], topdown_seg, ret['mu_td_s'], ret['logvar_td_s'], weight= weights['td_seg_channel_w'])
        td_seg_ext_loss = self.seg_loss(ret['topdown_rec_ext'], topdown_seg_ext, ret['mu_td_u'], ret['logvar_td_u'], weight= weights['td_ext_seg_channel_w'])
        tl_loss = self.tl_loss(ret['tl_indicator'], gt_traffic_light, weight=weights['tl_w'])
        ss_loss = self.tl_loss(ret['ss_indicator'], gt_stop_sign, weight=weights['ss_w'])
        # alignment_loss = self.contrastive_alignment_loss(ret['rgb_sf'], ret['lidar_sf'])
        alignment_loss = self.kl_contrastive_alignment_loss(ret['mu_front'], ret['logvar_front'], ret['mu_td_s'], ret['logvar_td_s'])
        losses = {
            'wp_loss': wp_loss,
            'wp_loss_zero': wp_loss_zero,
            'front_seg_loss': front_seg_loss,
            'td_seg_loss': td_seg_loss,
            'td_seg_ext_loss': td_seg_ext_loss,
            'tl_loss': tl_loss,
            'ss_loss': ss_loss,
            'alignment_loss': alignment_loss
        }
        return losses

    def forward(self, image_inputs, lidar_inputs, target_point, measurements):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_inputs (tensor): image_inputs with the shape of [B, C, H, W]
            lidar_list (tensor): lidar inputs with the shape of [B, C, H, W]
            target_point (tensor): goal location registered to ego-frame
        '''
        features = self.encoder(image_inputs, lidar_inputs, measurements)
        
        rgb_features = features[:,:512]
        lidar_features = features[:,512:1024]
        measurements = features[:, 1024:1028]

        rgb_shared_features = self.rgb_shared_features_extractor(rgb_features)
        rgb_unique_features = self.rgb_unique_features_extractor(rgb_features)

        lidar_shared_features = self.lidar_shared_features_extractor(lidar_features)
        lidar_unique_features = self.lidar_unique_features_extractor(lidar_features)

        z_td_s, mu_td_s, logvar_td_s = self._img_bottleneck_s(rgb_shared_features)
        z_td_u, mu_td_u, logvar_td_u = self._img_bottleneck_u(rgb_unique_features)

        z_front, mu_front, logvar_front = self._lidar_bottleneck(lidar_shared_features)

        front_rec = self.front_seg_decoder(z_front)
        topdown_rec = self.topdown_seg_decoder(z_td_s)
        topdown_rec_ext = self.topdown_extra_seg_decoder(z_td_u)

        tl_indicator = self.traffic_light_extractor(rgb_unique_features)
        ss_indicator = self.stop_sign_extractor(rgb_unique_features)

        mu_s, logvar_s = self._merge_gaussian(mu_td_s, logvar_td_s, mu_front, logvar_front)
        shared_features = self._reparameterize(mu_s, logvar_s)
        fused_features =  torch.cat([features, shared_features], dim=1)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        if not self.inference:
            ret = {
                'pred_wp': pred_wp,
                'front_rec': front_rec,
                'mu_front': mu_front, 
                'logvar_front': logvar_front, 
                'topdown_rec': topdown_rec,
                'topdown_rec_ext':topdown_rec_ext,
                'mu_td_s': mu_td_s, 
                'logvar_td_s': logvar_td_s, 
                'mu_td_u': mu_td_u, 
                'logvar_td_u': logvar_td_u, 
                'rgb_sf': rgb_shared_features,
                'lidar_sf': lidar_shared_features,
                'tl_indicator': tl_indicator,
                'ss_indicator': ss_indicator
                }
            return ret
        else:
            return pred_wp, tl_indicator, ss_indicator, front_rec

    def control_pid(self, waypoints, velocity, is_stuck):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
            is_stuck (bool): indicator for being stuck
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()
        if is_stuck:
            desired_speed = np.array(self.config.default_speed)
        else:
            desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
