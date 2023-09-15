#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
from tqdm import tqdm
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from render import render, render_waypoints

import torch
import torch.nn as nn
import torchvision.utils
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tensorboardX import SummaryWriter
import numpy as np
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.data import create_carla_dataset, create_carla_loader
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = ssl._create_unverified_context

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "--train-towns",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train towns (default: [0])",
)
parser.add_argument(
    "--val-towns",
    type=int,
    nargs="+",
    default=[1],
    help="dataset validation towns (default: [1])",
)
parser.add_argument(
    "--train-weathers",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train weathers (default: [0])",
)
parser.add_argument(
    "--val-weathers",
    type=int,
    nargs="+",
    default=[0],
    help="dataset validation weathers (default: [1])",
)
parser.add_argument(
    "--saver-decreasing",
    action="store_true",
    default=False,
    help="StarAt with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--with-lidar",
    action="store_true",
    default=False,
    help="load lidar data in the dataset",
)
parser.add_argument(
    "--with-seg",
    action="store_true",
    default=False,
    help="load segmentation data in the dataset",
)
parser.add_argument(
    "--with-depth",
    action="store_true",
    default=False,
    help="load depth data in the dataset",
)
parser.add_argument(
    "--multi-view",
    action="store_true",
    default=False,
    help="load multi-view data in the dataset",
)
parser.add_argument(
    "--multi-view-input-size",
    default=[3,128,128],
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224) for left- right- or rear- view",
)

parser.add_argument(
    "--freeze-num", type=int, default=-1, help="Number of freeze layers in the backbone"
)
parser.add_argument(
    "--backbone-lr", type=float, default=5e-4, help="The learning rate for backbone"
)
parser.add_argument(
    "--with-backbone-lr",
    action="store_true",
    default=False,
    help="The learning rate for backbone is set as backbone-lr",
)

# Dataset / Model parameters
parser.add_argument("data_dir", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="newcarla",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--model",
    default="resnet101",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "countception"',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--validation-batch-size-multiplier",
    type=int,
    default=1,
    metavar="N",
    help="ratio of validation batch size to training batch size (default: 1)",
)
parser.add_argument(
    "--augment-prob",
    type=float,
    default=0.0,
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd"',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)


# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=5e-6,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.9, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="const", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)"
)
parser.add_argument(
    "--smoothed_l1", type=bool, default=False, help="L1 smooth"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=5,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--name', type=str, default='loss_attack_interfuser')



class WaypointL1Loss:
    def __init__(self, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss(reduction="none")
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def __call__(self, output, target):
        invaild_mask = target.ge(1000)
        output[invaild_mask] = 0
        target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = 0
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def set_interfuser(args):

    # args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    # if "WORLD_SIZE" in os.environ:
    #     args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = 1 # torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        freeze_num=args.freeze_num,
    )

    checkpoint = torch.load(args.resume)
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k[7:] if k.startswith("module") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()


    return model

def retransform(data):
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    data = data * std_tensor + mean_tensor
    return data


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fns,
    args,
    writer,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_waypoints = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_junction = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    losses_stop_sign = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if isinstance(input, (tuple, list)):
            batch_size = input[0].size(0)
        elif isinstance(input, dict):
            batch_size = input[list(input.keys())[0]].size(0)
        else:
            batch_size = input.size(0)
        if not args.prefetcher:
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    target[key] = target[key].cuda()
            else:
                target = target.cuda()

        with amp_autocast():
            output = model(input)
            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
            loss_waypoints = loss_fns["waypoints"](output[1], target[1])
            loss_junction = loss_fns["cls"](output[2], target[2])
            on_road_mask = target[2] < 0.5
            loss_traffic_light_state = loss_fns["cls"](
                output[3], target[3]
            )
            loss_stop_sign = loss_fns["stop_cls"](output[4], target[6])
            loss = (
                loss_traffic * 0.5
                + loss_waypoints * 0.2
                + loss_velocity * 0.05
                + loss_junction * 0.05
                + loss_traffic_light_state * 0.1
                + loss_stop_sign * 0.01
            )

        if not args.distributed:
            losses_traffic.update(loss_traffic.item(), batch_size)
            losses_waypoints.update(loss_waypoints.item(), batch_size)
            losses_m.update(loss.item(), batch_size)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), batch_size)
                reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                losses_traffic.update(reduced_loss_traffic.item(), batch_size)
                reduced_loss_velocity = reduce_tensor(
                    loss_velocity.data, args.world_size
                )
                losses_velocity.update(reduced_loss_velocity.item(), batch_size)

                reduced_loss_waypoints = reduce_tensor(
                    loss_waypoints.data, args.world_size
                )
                losses_waypoints.update(reduced_loss_waypoints.item(), batch_size)
                reduced_loss_junction = reduce_tensor(
                    loss_junction.data, args.world_size
                )
                losses_junction.update(reduced_loss_junction.item(), batch_size)
                reduced_loss_traffic_light_state = reduce_tensor(
                    loss_traffic_light_state.data, args.world_size
                )
                losses_traffic_light_state.update(
                    reduced_loss_traffic_light_state.item(), batch_size
                )
                reduced_loss_stop_sign = reduce_tensor(
                    loss_stop_sign.data, args.world_size
                )
                losses_stop_sign.update(reduced_loss_stop_sign.item(), batch_size)
                if writer and args.local_rank == 0:
                    writer.add_scalar("train/loss", reduced_loss.item(), num_updates)
                    writer.add_scalar(
                        "train/loss_traffic", reduced_loss_traffic.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_velocity", reduced_loss_velocity.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_waypoints",
                        reduced_loss_waypoints.item(),
                        num_updates,
                    )
                    writer.add_scalar(
                        "train/loss_junction", reduced_loss_junction.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_traffic_light_state",
                        reduced_loss_traffic_light_state.item(),
                        num_updates,
                    )
                    writer.add_scalar(
                        "train/loss_stop_sign",
                        reduced_loss_stop_sign.item(),
                        num_updates,
                    )

                    # Add Image
                    writer.add_image(
                        "train/front_view", retransform(input["rgb"][0]), num_updates
                    )
                    writer.add_image(
                        "train/left_view",
                        retransform(input["rgb_left"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/right_view",
                        retransform(input["rgb_right"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/front_center_view",
                        retransform(input["rgb_center"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_traffic",
                        torch.clip(output[0][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0],
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_traffic_render",
                        torch.clip(
                            torch.tensor(
                                render(
                                    output[0][0].view(20, 20, 7).detach().cpu().numpy()
                                )[:100, 40:140]
                            ),
                            0,
                            255,
                        ).view(1, 100, 100),
                        num_updates,
                    )
                    input["lidar"][0] = input["lidar"][0] / torch.max(input["lidar"][0])
                    writer.add_image(
                        "train/lidar", torch.clip(input["lidar"][0], 0, 1), num_updates
                    )
                    writer.add_image(
                        "train/gt_traffic",
                        torch.clip(target[4][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0],
                        num_updates,
                    )
                    writer.add_image(
                        "train/gt_highres_traffic",
                        torch.clip(target[0][0], 0, 1),
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_waypoints",
                        torch.clip(
                            torch.tensor(
                                render_waypoints(output[1][0].detach().cpu().numpy())[
                                    :100, 40:140
                                ]
                            ),
                            0,
                            255,
                        ).view(1, 100, 100),
                        num_updates,
                    )
                    writer.add_image(
                        "train/gt_waypoints",
                        torch.clip(target[5][0], 0, 1),
                        num_updates,
                    )

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss(traffic): {loss_traffic.val:>9.6f} ({loss_traffic.avg:>6.4f})  "
                    "Loss(waypoints): {loss_waypoints.val:>9.6f} ({loss_waypoints.avg:>6.4f})  "
                    "Loss(junction): {loss_junction.val:>9.6f} ({loss_junction.avg:>6.4f})  "
                    "Loss(light): {loss_traffic_light_state.val:>9.6f} ({loss_traffic_light_state.avg:>6.4f})  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        loss_traffic=losses_traffic,
                        loss_waypoints=losses_waypoints,
                        loss_junction=losses_junction,
                        loss_traffic_light_state=losses_traffic_light_state,
                        batch_time=batch_time_m,
                        rate=batch_size * args.world_size / batch_time_m.val,
                        rate_avg=batch_size * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    model, loader, loss_fns, args, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_waypoints = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_junction = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    losses_stop_sign = AverageMeter()

    l1_errorm = AverageMeter()
    junction_errorm = AverageMeter()
    traffic_light_state_errorm = AverageMeter()
    stop_sign_errorm = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if isinstance(input, (tuple, list)):
                batch_size = input[0].size(0)
            elif isinstance(input, dict):
                batch_size = input[list(input.keys())[0]].size(0)
            else:
                batch_size = input.size(0)
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    input[key] = input[key].cuda()
            else:
                target = target.cuda()

            with amp_autocast():
                output = model(input)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
            loss_waypoints = loss_fns["waypoints"](output[1], target[1])
            loss_junction = loss_fns["cls"](output[2], target[2])
            on_road_mask = target[2] < 0.5
            loss_traffic_light_state = loss_fns["cls"](output[3], target[3])
            loss_stop_sign = loss_fns["stop_cls"](output[4], target[6])
            loss = (
                loss_traffic * 0.5
                + loss_waypoints * 0.2
                + loss_velocity * 0.05
                + loss_junction * 0.05
                + loss_traffic_light_state * 0.1
                + loss_stop_sign * 0.01
            )


    return loss



def plot_img_from_normalized_img(img_array, is_normalized=True):
    img_to_be_plotted = copy.deepcopy(img_array)
    assert len(img_array.shape) == 3
    if img_to_be_plotted.shape[0] == 3:
        img_to_be_plotted = img_to_be_plotted.transpose(1, 2, 0)
    if is_normalized:
        for idx, (m, v) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            img_to_be_plotted[:, :, idx] = (img_to_be_plotted[:, :, idx] * v) + m
    return img_to_be_plotted


class ImageDot_front(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot_front, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha =  nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0]),requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0]),requires_grad=True)
        self.beta =  nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0]),requires_grad=True)
        self.center = nn.Parameter(torch.tensor([
            [0.25, 0.25],  [0.25, 0.75],
            
            [0.75, 0.25],  [0.75, 0.75]]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            [0.5, 0.5, 0.5],  [0.5, 0.5, 0.5],
            
            [0.5, 0.5, 0.5],  [0.5, 0.5, 0.5]]),
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
class ImageDot_left(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot_left, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha =  nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.beta =  nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.center = nn.Parameter(torch.tensor([
            
            [0.5, 0.25], [0.5, 0.75],
            ]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            ]),
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
class ImageDot_right(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot_right, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha =  nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.beta =  nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)
        self.center = nn.Parameter(torch.tensor([
            
            [0.5, 0.25], [0.5, 0.75],
            ]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            ]),
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
class ImageDot_front_center(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot_front_center, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha =  nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.beta =  nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.center = nn.Parameter(torch.tensor([
           
            [0.5, 0.5]
            ]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            
           [0.5, 0.5, 0.5]
            ]),
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
    def __init__(self,args):
        super(AttackModel, self).__init__()
        self.image_dot_front = ImageDot_front()
        self.image_dot_left = ImageDot_left()
        self.image_dot_right = ImageDot_right()
        self.image_dot_front_center = ImageDot_front_center()
        # self.base_model = models.resnet50(pretrained=True).eval()
        model = set_interfuser(args).to(args.device)
        # model.load_state_dict(torch.load(os.path.join(args.logdir,'best_model.pth')))
        self.base_model = model.eval()

        # self.base_model = model


        self.device = args.device
        self.args = args
        self._freeze_pretrained_model()


    def _freeze_pretrained_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self,x):
        front_image = x["rgb"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        front_center_image = x["rgb_center"]

        front_image = self.image_dot_front(front_image)
        left_image = self.image_dot_left(left_image)
        right_image = self.image_dot_right(right_image)
        front_center_image  = self.image_dot_front_center(front_center_image)

        x["rgb"] = front_image 
        x["rgb_left"] = left_image
        x["rgb_right"] = right_image
        x["rgb_center"] = front_center_image


        return self.base_model(x)


def create_tqdm_bar(iterable,desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc, leave=False)

def train(model, train_loader,tb_logger, args, num_of_runs, loss_fns):
    epochs = args.epochs
    name = args.name

    training_loss = 0
    lr = 0.0008

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        if (epoch + 1) % 2 == 0:
            lr *= 0.5
        model.zero_grad()
        training_loop = create_tqdm_bar(train_loader, desc=f"Training Epoch [{epoch}/{epochs}]")
        # for train_iteration, data in training_loop:
        for batch_idx,(input, target) in training_loop:

            if isinstance(input, (tuple, list)):
                batch_size = input[0].size(0)
            elif isinstance(input, dict):
                batch_size = input[list(input.keys())[0]].size(0)
            else:
                batch_size = input.size(0)
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    input[key] = input[key].cuda()
            else:
                target = target.cuda()

            # with amp_autocast():
            output = model(input)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
            loss_waypoints = loss_fns["waypoints"](output[1], target[1])
            loss_junction = loss_fns["cls"](output[2], target[2])
            on_road_mask = target[2] < 0.5
            loss_traffic_light_state = loss_fns["cls"](output[3], target[3])
            loss_stop_sign = loss_fns["stop_cls"](output[4], target[6])
            loss = (
                loss_traffic * 0.5
                + loss_waypoints * 0.2
                + loss_velocity * 0.05
                + loss_junction * 0.05
                + loss_traffic_light_state * 0.1
                + loss_stop_sign * 0.01
            )
            # create batch and move to GPU


            model_loss = - loss
            # torch.backends.cudnn.enabled=False
            model_loss.backward(retain_graph=True)
            # optimizer.step()
            nn.utils.clip_grad_norm(model.parameters(),max_norm=10.0,norm_type=2.0)
            for param in model.parameters():
                if param.requires_grad == True:
                    # nn.utils.clip_grad_norm(param,max_norm=1.0,norm_type=2.0)
                    param.data = torch.clamp(param.data - param.grad.data * lr, min=0.0, max=1.0)
            if epoch == epochs - 1:
                training_loss += model_loss.item()

            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(model_loss.item()))
            tb_logger.add_scalar(f'{name}/train_loss', model_loss.item(), epoch * len(train_loader) + batch_idx)


        if epoch != 0 and epoch % 10 ==0:
            # save_dir = os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_ckpt.tar')
            torch.save(model.image_dot_front.state_dict(),os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_front_ckpt.tar'))
            torch.save(model.image_dot_left.state_dict(),os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_left_ckpt.tar'))
            torch.save(model.image_dot_right.state_dict(),os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_right_ckpt.tar'))
            torch.save(model.image_dot_front_center.state_dict(),os.path.join('logs',name, f'run_{num_of_runs}',f'{epoch}_center_ckpt.tar'))

            # load ckpt
            # model_dict = torch.load('logs/stop_sign_attack/run_29/ckpt.tar')
            # model = ImageDot()
            # model.load_state_dict(model_dict)
            # test = model(test.unsqueeze(0))

   


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    args, args_text = _parse_args()
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.world_size = 1
    args.rank = 0

    data_config = resolve_data_config(
        vars(args), verbose=args.local_rank == 0)

    dataset_eval = create_carla_dataset(
        args.dataset,
        root=args.data_dir,
        towns=args.val_towns,
        weathers=args.val_weathers,
        batch_size=args.batch_size,
        with_lidar=args.with_lidar,
        with_seg=args.with_seg,
        with_depth=args.with_depth,
        multi_view=args.multi_view,
        augment_prob=args.augment_prob
        )

    collate_fn = None
    mixup_fn = None

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_eval = create_carla_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        multi_view_input_size=args.multi_view_input_size,
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
        )

    # setup loss function
    if args.smoothing > 0:
        cls_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        cls_loss = nn.CrossEntropyLoss()

    if args.smoothed_l1:
        l1_loss = torch.nn.SmoothL1Loss
    else:
        l1_loss = torch.nn.L1Loss


    validate_loss_fns = {
        "traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "waypoints": WaypointL1Loss(l1_loss=l1_loss),
        "cls": cls_loss,
        "stop_cls": cls_loss,
        }

    model = AttackModel(args)

    path = os.path.join('logs',args.name)
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path,f'run_{num_of_runs+1}')
    tb_logger = SummaryWriter(path)

    train(model,loader_eval,tb_logger,args,num_of_runs+1,validate_loss_fns)

