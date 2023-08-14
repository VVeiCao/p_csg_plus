import argparse
import json
import os
import pdb
from matplotlib import image
from tqdm import tqdm

import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from conf.leaderboard_config import GlobalConfig
from models.p_csg import P_CSG
from dataset.data_new import CARLA_Data
from utils.utils import calculate_penalty, log_name

import pdb
# import ssl


# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_contex
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='p-csg', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
parser.add_argument('--save_all_models', action='store_true', default=False, help='if save all models during the training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--logdir', type=str, default='output/', help='Directory to log data to.')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to use')
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
parser.add_argument('--speed_lb_ss', type= float, default=1.5, help='speed lower bound to avoid stop sign punishment')
parser.add_argument('--save_program_files', action='store_true', default=False, help='flag for saving program files in experiment path before each training')
parser.add_argument('--warm_up', type=int, default=20, help='warm up epochs')
args = parser.parse_args()

# Config
config = GlobalConfig()
group, time = log_name()
if args.logdir == 'output/':
	args.logdir = os.path.join(args.logdir, group, time)


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.extent = (2.450842, 1.064162, 0.755373) # (1 / 2 * （lenght, width, height）)
		r = np.pi / 2
		c, s = np.cos(r), np.sin(r)
		r_matrix = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
		self.rotation_matrix_pi2 = torch.from_numpy(r_matrix).to(args.device, dtype=torch.float32)

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()

		# Train loop
		for data in tqdm(dataloader_train):
			
			# efficiently zero gradients
			for p in model.parameters():
				p.grad = None
			
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
			ret = model(fronts, lidars, target_point, gt_measurements)
			losses = model.losses(
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
				red_light_penalty = calculate_penalty(ret['pred_wp'], light, self.rotation_matrix_pi2, data, self.extent,is_pred_waypoints=True).mean()
				red_light_penalty_gt = calculate_penalty(gt_waypoints, light, self.rotation_matrix_pi2, data, self.extent)
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
			
			loss.backward()
			loss_epoch += float(loss.item())

			num_batches += 1
			optimizer.step()

			writer.add_scalar('train_loss', loss.item(), self.cur_iter)
			writer.add_scalar('red_light_penalty', red_light_penalty.item(), self.cur_iter)
			writer.add_scalar('traffic_light_indicator_loss', tl_loss.item(), self.cur_iter)
			writer.add_scalar('stop_sign_indicator_loss', ss_loss.item(), self.cur_iter)
			writer.add_scalar('tp_seg_loss', td_seg_loss.item(), self.cur_iter)
			writer.add_scalar('front_seg_loss', front_seg_loss.item(), self.cur_iter)
			writer.add_scalar('alignment_loss', alignment_loss.item(), self.cur_iter )
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self):
		model.eval()

		with torch.no_grad():	
			num_batches = 0
			val_loss_epoch = 0.
			red_light_penalty_epoch = 0.
			speed_penalty_epoch = 0.
			stop_sign_penalty_epoch = 0.
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				
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
				
				# traffic light information
				light = data["light"].to(args.device, dtype=torch.float32)
				stop_sign = data["stop_sign"].to(args.device, dtype=torch.bool)

				ret = model(fronts, lidars, target_point, gt_measurements)

				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				wp_loss = model.wp_loss(ret['pred_wp'], gt_waypoints, reduction='none').mean(dim=[1,2])
				wp_loss_zero = model.wp_loss_zero(ret['pred_wp'], reduction='none').mean(dim=[1,2])

				if (torch.sum(light == 23) != 0 or torch.sum(light == 24) != 0):
					red_light = torch.logical_or((light == 23), (light == 24))
					red_light_penalty = calculate_penalty(ret['pred_wp'], light, self.rotation_matrix_pi2, data, self.extent, is_pred_waypoints=True).mean()
					red_light_penalty_gt = calculate_penalty(gt_waypoints, light, self.rotation_matrix_pi2, data, self.extent)
					red_light_violation = red_light_penalty_gt > 0
					
					loss_pos_obey_rules = torch.cat((wp_loss[~red_light], wp_loss[red_light][~red_light_violation], wp_loss_zero[red_light][red_light_violation]))
					loss = args.lambda1 * red_light_penalty + loss_pos_obey_rules.mean() if loss_pos_obey_rules.nelement() != 0 else args.lambda1 * red_light_penalty
				else:
					red_light_penalty = torch.tensor(0.0)
					loss = wp_loss.mean()		

				# speed penalty calculation
				desired_speed = torch.linalg.norm(ret['pred_wp'][:, 0, :] - ret['pred_wp'][:, 1, :], dim=-1) * 2.0
				d_theta = gt_theta[:, -1] - gt_theta[:, 0]
				speed_penalty = torch.abs(torch.sin(d_theta)) * (torch.clamp(desired_speed, min=args.speed_lb) - args.speed_lb)
				speed_penalty = speed_penalty.mean()
				loss = loss + args.lambda2 * speed_penalty

				# stop sign penalty calculation
				desired_speed = torch.linalg.norm(ret['pred_wp'][:, 0, :] - ret['pred_wp'][:, 1, :], dim=-1) * 2.0
				if stop_sign.any():
					stop_sign_penalty = stop_sign.float() * torch.clamp(desired_speed - args.speed_lb_ss, min=0.0)
					stop_sign_penalty = stop_sign_penalty.mean()
				else:
					stop_sign_penalty = torch.tensor(0.0)

				loss = loss + args.lambda3 * stop_sign_penalty

				val_loss_epoch += float(loss.item())
				red_light_penalty_epoch += float(red_light_penalty.item())
				speed_penalty_epoch += float(speed_penalty.item())
				stop_sign_penalty_epoch += float(stop_sign_penalty.item())
				num_batches += 1
					
			val_loss = val_loss_epoch / float(num_batches)
			red_light_penalty = red_light_penalty_epoch / float(num_batches) 
			speed_penalty_epoch = speed_penalty_epoch / float(num_batches) 
			stop_sign_penalty_epoch = stop_sign_penalty_epoch /float(num_batches)
			#l2loss = l2epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {val_loss:3.3f}') #+ f' l2loss: {l2loss:3.3f}')

			writer.add_scalar('val_loss', val_loss, self.cur_epoch)
			writer.add_scalar('val_red_light_penalty', red_light_penalty, self.cur_epoch)
			writer.add_scalar('val_speed_penalty', speed_penalty_epoch, self.cur_epoch)
			writer.add_scalar('val_stop_sign_penalty', stop_sign_penalty_epoch, self.cur_epoch)
			self.val_loss.append(val_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			}

		# Save ckpt for every epoch
		if args.save_all_models:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

# Data
train_set = CARLA_Data(root=config.train_data, config=config)
val_set = CARLA_Data(root=config.val_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = P_CSG(config).to(args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)
# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir, exist_ok=True)
	print('Created dir:', args.logdir)
	writer = SummaryWriter(log_dir=args.logdir)
	# save program files
	if args.save_program_files:
		shutil.copy('team_code/train.py', args.logdir)
		shutil.copytree('team_code/conf/', os.path.join(args.logdir, 'conf/'))
		shutil.copytree('team_code/dataset/', os.path.join(args.logdir, 'dataset/'))
		shutil.copytree('team_code/models/', os.path.join(args.logdir, 'models/'))
	# Load checkpoint if specified
	if args.ckpt and os.path.isfile(os.path.join(args.ckpt, 'recent.log')):
		print('Loading checkpoint from ' + args.ckpt)
		with open(os.path.join(args.ckpt, 'recent.log'), 'r') as f:
			log_table = json.load(f)
		
		# Load checkpoint
		model.load_state_dict(torch.load(os.path.join(args.ckpt, 'best_model.pth')))
		optimizer.load_state_dict(torch.load(os.path.join(args.ckpt, 'best_optim.pth')))
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	writer = SummaryWriter(log_dir=args.logdir)
	print('Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	trainer.bestval_epoch = log_table['bestval_epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))
else:
	print('No recent.log files! Check your log directory')


# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

with open(os.path.join(args.logdir, 'conf.txt'), 'w') as f:
	json.dump(config.__dict__, f, indent=2)
	
for epoch in range(trainer.cur_epoch, args.epochs): 
	trainer.train()
	if epoch % args.val_every == 0 and epoch >= args.warm_up: 
		trainer.validate()
		trainer.save()