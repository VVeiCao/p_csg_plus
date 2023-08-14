import numpy as np
from matplotlib import pyplot as plt
import argparse
import json
import os
import pdb
parser = argparse.ArgumentParser()

parser.add_argument('--logdir', type=str, default='output/log_vae_cross_l2loss/cross_vae_bce_gray_img_feature_alignment(rgb_range_0_1)(rgb_lidar_input_aligned)image_vae_w_1.0lidar_vae_w_1.0feature_alignment_weight_0.01')
parser.add_argument('--logfile', type=str, default='recent.log')
parser.add_argument('--savedir', type=str, default='figures')
args = parser.parse_args()
# open result file
path = os.path.join(args.logdir, args.logfile)
f = open(path)

data = json.load(f)
train_loss = data['train_loss']
val_loss = data['val_loss']

x = np.arange(len(val_loss))
y = val_loss

plt.xlabel('epoch')
plt.ylabel('loss')

plt.plot(x, y)
plt.savefig(os.path.join(args.logdir, 'val_loss'))

f.close()
