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
import argparse
import time

import numpy as np
import pandas as pd
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils

import matplotlib.pyplot as plt


from my_timm import create_dataset, create_dataset_from_file, create_loader, Mixup, FastCollateMixup
from timm.data import resolve_data_config, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.models.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

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

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')




from cli import parse_commandline_args


def main(args):
    #pre-setup


    args = setup(args)
    random_seed(args.seed)

    torch.cuda.set_device(args.device_id)
    print(torch.cuda.get_device_name())
    # init model, optmizizer, datasets, etc
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
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

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not args.use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    if args.grad_accumulation>1:
        assert args.use_amp is None, 'Cannot use AMP with gradient accumulation'

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after to(device), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    _logger.info('Scheduled epochs: {}'.format(num_epochs))

    print(f'dataset: {args.dataset}')
    dataset_train = create_dataset_from_file(args.dataset, args.data_dir, args.data_file_train)
    dataset_eval = create_dataset_from_file(args.dataset, args.data_dir, args.data_file_val)


    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        rot=args.rot,
        hflip=args.hflip,
        vflip=args.vflip,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

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

    # setup loss function
    if args.bce:
        train_loss_fn = BinaryCrossEntropy()
    # elif mixup_active:
    #     passs
    #     # smoothing is handled with mixup target transform which outputs sparse, soft targets
    #     #train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
    else:
        raise NotImplementedError


    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = BinaryCrossEntropy().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
    args.output = get_outdir(args.output if args.output else '../output/train', exp_name)
    output_dir = args.output
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):

            if args.sched=='plateau':
                lr = optimizer.param_groups[0]['lr']
                if lr<1e-6:
                    _logger.info('lr: {} for plateau scheduler is below 1e-6'.format(lr))
                    break

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:

                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

                try:
                    df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
                    df.plot(subplots=True)
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(output_dir, 'summary_plot.png'))
                except:
                    pass

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    return args


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target.float())
            loss /= args.grad_accumulation # virtually increase batchsize by gradient accumulation


        losses_m.update(loss.item(), input.size(0))

        batch_accumulation_full = ((batch_idx + 1) % args.grad_accumulation == 0) or (batch_idx + 1 == len(loader))

        if loss_scaler is not None:
            optimizer.zero_grad()
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)

            if batch_accumulation_full:
                #print('bacth_accumulation is full')
                optimizer.step()
                optimizer.zero_grad()

        if not batch_accumulation_full:
            #print('bacth_accumulation is not full')
            continue # only do code after this is batch accumulator is full

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)





        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg), ('lr', optimizer.param_groups[0]['lr'])])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    # top1_m = AverageMeter()
    # top5_m = AverageMeter()

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

            loss = loss_fn(output, target.float())
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            target_list.extend(target.detach().cpu().numpy())
            output_list.extend(output.detach().cpu().numpy())

            reduced_loss = loss.data
            #torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            # top1_m.update(acc1.item(), output.size(0))
            # top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m))

    from sklearn import metrics
    targets = np.asarray(target_list)
    outputs = np.asarray(output_list)
    auc_roc = metrics.roc_auc_score(targets, outputs)
    auc_pr = metrics.average_precision_score(targets, outputs)

    metrics = OrderedDict([('loss', losses_m.avg), ('auc_roc', auc_roc), ('auc_pr', auc_pr)])
    print(metrics)

    return metrics

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