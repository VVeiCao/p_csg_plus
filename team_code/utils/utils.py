import torch
import numpy as np

from conf.config import GlobalConfig
from datetime import date
from time import gmtime, strftime

config = GlobalConfig()

def calculate_penalty(waypoints, light, rotation_matrix_pi2, data, extent, device='cuda', is_pred_waypoints = False):
	rotation_matrix = data["rotation_matrix"].to(device, dtype=torch.float32)
	B = waypoints.shape[0]
	ones = torch.ones(B, waypoints.shape[1], 1).to(device, dtype=torch.float32)
	pred_wp_plus = torch.cat([waypoints, ones], dim=-1)

	red_light = torch.logical_or((light == 23), (light == 24))
	pred_wp_plus = pred_wp_plus[red_light]
	rotation_matrix = rotation_matrix[red_light]
	nB = torch.sum(red_light)
	rotation_matrix = rotation_matrix.unsqueeze(1).repeat(1, config.pred_len, 1, 1).view(nB * config.pred_len,
                                                                                        3, 3)
	pred_wp_plus = pred_wp_plus.view(nB * config.pred_len, 3, 1)
	pred_wp_world = torch.bmm(rotation_matrix, pred_wp_plus)
	rotation_matrix_pi2 = rotation_matrix_pi2.squeeze(0).repeat(nB * config.pred_len, 1, 1)
	pred_wp_world = torch.bmm(rotation_matrix_pi2, pred_wp_world).view(nB, config.pred_len, 3)[:, :, :2]

	line_pos = data["value"].to(device, dtype=torch.float32)[red_light]
	direction_1 = data["direction_1"].to(device, dtype=torch.long)[red_light]
	direction_2 = data["direction_2"].to(device, dtype=torch.float32)[red_light]
	pred_wp_world_xy = torch.stack([it[:, d1] for (it, d1) in zip(pred_wp_world, direction_1)], dim=0)

	dis = (pred_wp_world_xy - line_pos.unsqueeze(1)) * direction_2.unsqueeze(1) + extent[0]

	upper_bound = 5
	if is_pred_waypoints:
		xy = data["xy"].to(device, dtype=torch.float32)[red_light]
		loc_world_xy = torch.stack([loc[d1] for (loc, d1) in zip(xy, direction_1)], dim=0)
		dis_loc = (loc_world_xy - line_pos) * direction_2
		in_area = dis_loc > 0.0
		dis = torch.clamp(dis - (in_area * dis_loc).unsqueeze(-1).repeat(1, 4), 0, upper_bound) ** 2
	else:
		# dis_loc = None
		dis = torch.clamp(dis, 0, upper_bound) ** 2

	coeff = config.penalty_w
	coeff = coeff / np.sum(coeff)
	coeff = torch.FloatTensor(coeff).to(device, dtype=torch.float32)
	red_light_penalty = torch.mean(coeff.unsqueeze(0).repeat(nB, 1) * dis, dim=1)
	return red_light_penalty

def log_name():
    group = str(date.today().strftime("%d_%m_%Y"))
    time = str(strftime("%H:%M:%S", gmtime()))
    return group, time

