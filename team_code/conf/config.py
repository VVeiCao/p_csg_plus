import json
import os

class GlobalConfig:

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __init__(self, **kwargs):
        """ base architecture configurations """
        # Data
        self.seq_len = 1 # input timesteps
        self.pred_len = 4 # future waypoints predicted

        self.root_dir = 'carla_data/clear-weather/data'
        # self.train_towns = ['Town07', 'Town10']
        self.train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
        self.val_towns = ['Town05']
        self.train_data, self.val_data = [], []
        for town in self.train_towns:
            self.train_data.append(os.path.join(self.root_dir, town+'_short'))
            self.train_data.append(os.path.join(self.root_dir, town+'_tiny'))
            self.train_data.append(os.path.join(self.root_dir, town+'_long'))
        for town in self.val_towns:
            self.val_data.append(os.path.join(self.root_dir, town+'_short'))
            self.val_data.append(os.path.join(self.root_dir, town+'_long'))
            self.val_data.append(os.path.join(self.root_dir, town+'_tiny'))


        # Lidar 
        self.pixels_per_meter = 8
        self.hist_max_per_pixel = 5
        self.x_meters_max = 16
        self.y_meters_max = 32

        # Segmentation 
        self.front_channel_weight = [5, 1, 1] # obstale|road|others
        self.td_channel_weight = [5, 1, 1] # obstale|road|others
        self.td_ext_channel_weight = [3,1] # road_line|red_light|green_light|stop_sign|others
        self.tl_weight = [4,8,1] # red/yellow, green, none
        self.ss_weight = [5,1] # the weight of the stop sign
        self.penalty_w = [4.0, 3.0, 2.0, 1.0]
        self.KL_weight = 5e-4

        self.input_resolution = [256,256]

        self.scale = 1 # image pre-processing
        self.crop = 256 # image pre-processing

        self.td_scale = 1.0 # segmentation pre-processing

        self.lr = 1e-4 # learning rate
        # Image Encoder
        self.image_encoder_model = 'resnet50' # resnet50
        self.lidar_encoder_model = 'resnet18' 
        self.c_dim = 512
        self.lidar_in_channels = 2
        self.pretrained = True

        # Conv Encoder
        self.vert_anchors = 8
        self.horz_anchors = 8
        self.anchors = self.vert_anchors * self.horz_anchors

        # loss criterion
        self.waypoint_loss_criterion = 'l1_loss' #  huber_loss|mse_loss
        # GPT Encoder
        self.n_embd = 512
        self.block_exp = 4
        self.n_layer = 8
        self.n_head = 4
        self.n_scale = 4
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1

        # Controller
        self.turn_KP = 1.25
        self.turn_KI = 0.75
        self.turn_KD = 0.3
        self.turn_n = 20 # buffer size

        self.speed_KP = 5.0
        self.speed_KI = 0.5
        self.speed_KD = 1.0
        self.speed_n = 20 # buffer size

        self.max_throttle = 0.75 # upper limit on throttle signal value in dataset
        self.brake_speed = 0.1 # desired speed below which brake is triggered
        self.brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
        self.clip_delta = 0.25 # maximum change in speed input to logitudinal controller

        self.action_repeat = 2
        self.stuck_threshold = 1100 / self.action_repeat  # Number of frames after which the creep controller starts triggering. Divided by
        self.block_threshold = 2200 / self.action_repeat
        self.creep_duration = 30 / self.action_repeat  # Number of frames we will creep forward
        self.default_speed = 4.0  # Speed used when creeping

        # Size of the safety box
        self.safety_box_z_min = -2.0
        self.safety_box_z_max = -1.05

        self.safety_box_y_min = -4.0
        self.safety_box_y_max = 0.0

        self.safety_box_x_min = -1.066
        self.safety_box_x_max = 1.066
        self.safety_box_n = 50     # the minimum number of points in safety_box indicating objects front.
        self.save_frames = False  # if save frames, during the evaluation.
        self.save_frequency = 5

        for k,v in kwargs.items():
            setattr(self, k, v)
