# !/usr/bin/env python3
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

import sys
sys.path.append('../')

import argparse
import glob
import time

import numpy as np
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils

from my_timm import create_dataset, create_dataset_from_file, create_loader
from timm.data import resolve_data_config
from timm.models import create_model, safe_model_name
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from metrics import roc_pr_score, roc_auc_score, ece_score

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False



os.environ['CUDA_VISIBLE_GPU'] = 'cuda:1'
torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('test')


#todo:

# model:
# args.pretrained
# args.initial_checkpoint
# args.num_classes

from cli import parse_commandline_args
import csv


def main(args):
    #pre-setup

    args = setup(args)
    random_seed(args.seed)

    # init model, optmizizer, datasets, etc
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        # drop_rate=args.drop,
        # drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        # drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # move model to GPU, enable channels last layout if set
    model.cuda()


    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')
        loss_scaler = ApexScaler()
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        _logger.info('AMP not enabled. Training in float32.')


    dataset_eval = create_dataset_from_file(args.dataset, args.data_dir, args.data_file_val)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )


    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
    best_epoch = checkpoint['epoch']


    exp_name = '-'.join([args.experiment,
        safe_model_name(args.model),
        str(data_config['input_size'][-1])
    ])


    output_dir = get_outdir(args.output if args.output else '../output/test', exp_name)
    args.output_dir = output_dir
    print(f'outputdir: {args.output_dir}')
    auc_roc, auc_pr, ece = validate(model, loader_eval, args, amp_autocast=amp_autocast)

    eval_metrics = {'mean': {'auc_roc': float(auc_roc[0]), 'auc_pr': float(auc_pr[0]), 'ece': float(ece[0])},
                    'class_wise': {'auc_roc': [float(x) for x in auc_roc[1]],
                                    'auc_pr': [float(x) for x in auc_pr[1]],
                                    'ece': [float(x) for x in ece[1]]},
                    'best_epoch' : best_epoch,
                    'model': args.model,
                    'checkpoint' : args.initial_checkpoint
                    }

    auc_roc2 = dict(zip(loader_eval.dataset.classes, auc_roc[1]))
    auc_pr2 = dict(zip(loader_eval.dataset.classes, auc_pr[1]))
    ece2 = dict(zip(loader_eval.dataset.classes, ece[1]))

    rowd = OrderedDict(model=args.model, experiment=args.experiment, best_epoch=best_epoch, batch_size=args.batch_size*args.grad_accumulation)
    rowd.update([('roc_mean', auc_roc[0])])
    rowd.update([('pr_mean', auc_pr[0])])
    rowd.update([('ece_mean', ece[0])])


    rowd.update([ ('roc_'+ k.replace(" ", "_"), v) for k,v in auc_roc2.items() ])
    rowd.update([ ('pr_'+ k.replace(" ", "_"), v) for k,v in auc_pr2.items() ])
    rowd.update([ ('ece_'+ k.replace(" ", "_"), v) for k,v in ece2.items() ])

    filename = os.path.join(os.path.split(output_dir)[0],  'eval_summary.csv')
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        # if write_header:  # first iteration (epoch == 1 can't be used)
        #     dw.writeheader()
        dw.writeheader()
        dw.writerow(rowd)




    # eval_metrics = {'best_epoch':best_epoch,
    #                 'model': args.model,
    #                 'checkpoint' : args.initial_checkpoint,
    #                 **eval_metrics}

    # eval_metrics['best_epch'] = best_epoch
    #print(eval_metrics)


    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        f.write(args_text)
    # with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
    #     #doc = yaml.dump(eval_metrics, f)
    #     # Cache the args as a text string to save them in the output dir later
    #     args_text = yaml.safe_dump(eval_metrics, default_flow_style=False)
    #     f.write(args_text)

    # rowd = OrderedDict(epoch=epoch)
    # rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    # rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    # if log_wandb:
    #     wandb.log(rowd)
    # with open(filename, mode='a') as cf:
    #     dw = csv.DictWriter(cf, fieldnames=rowd.keys())
    #     if write_header:  # first iteration (epoch == 1 can't be used)
    #         dw.writeheader()
    #     dw.writerow(rowd)


    _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))



def validate(model, loader, args, amp_autocast=suppress, log_suffix=''):

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    target_list = []
    output_list = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            output = torch.sigmoid(output)

            target_list.extend(target.detach().cpu().numpy())
            output_list.extend(output.detach().cpu().numpy())



    targets = np.asarray(target_list)
    outputs = np.asarray(output_list)



    auc_roc = roc_auc_score(targets, outputs)
    auc_pr = roc_pr_score(targets, outputs)
    ece = ece_score(targets, outputs, loader.dataset.classes, loader.dataset.class_to_idx, plot_ece=True, save_path=args.output_dir)
    #
    # out = {'auc_roc': dict(**{'mean': auc_roc[0]}, **dict(zip(loader.dataset.classes, auc_roc[1]))),
    #         'auc_pr': dict(**{'mean': auc_pr[0]}, **dict(zip(loader.dataset.classes, auc_pr[1]))),
    #         'ece': dict(**{'mean': ece[0]}, **dict(zip(loader.dataset.classes, ece[1])))}

    return auc_roc, auc_pr, ece

def setup(args):
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher

    args.world_size = 1
    args.rank = 0  # global rank
    args.distributed=False
    _logger.info('Training with a single process on 1 GPUs.')

    # resolve AMP arguments based on PyTorch / Apex availability
    args.use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        args.use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        args.use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    # if args.fuser:
    #     set_jit_fuser(args.fuser)

    return args


if __name__ == '__main__':
    # parse arguments

    setup_default_logging()
    args = parse_commandline_args()
    main(args)
