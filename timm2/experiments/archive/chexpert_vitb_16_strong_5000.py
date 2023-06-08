import train_timm_multilabel
from cli import parse_dict_args
from timm.utils import *

def parameters(): 
    defaults = {
        'dataset': 'chexpert',
        'num-classes': 5,
        'data-dir': '/home/medssl/data/',

        'crop-pct': 1, # for validation only
        'mean': 0.5,
        'std': 0.5,
        'interpolation': '', #help = 'Image resize interpolation type (overrides model)')

        'epoch-repeats': 0.,
        # help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
        'start-epoch': 0,  # help='manual epoch number (useful on restarts)')


    # Model parameters

        'pretrained': 'true',
        #'gp': None #help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
        #'img-size': None #help='Image patch size (default: None => model default)')
        #'input-size': None, #nargs=3, type=int, metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
        'drop': 0.0,  # metavar='PCT', help='Dropout rate (default: 0.)')
        #'drop-connect': None,  # metavar='PCT', help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
        #'drop-path': None,  # metavar='PCT', help='Drop path rate (default: None)')
        #'drop-block': None,  # type=float,  metavar='PCT', help='Drop block rate (default: None)')

        # Optimizer parameters
        'opt': 'sgd',
        'momentum': 0.9, #help='Optimizer momentum (default: 0.9)')
        'weight-decay': 0.03, #help='weight decay (default: 2e-5)')
        # 'opt-eps': None, #help='Optimizer Epsilon (default: None, use opt default)')
        #'opt-betas': None, #type=float, help='Optimizer Betas (default: None, use opt default)')
        'clip-grad': 1, #help='Clip gradient norm (default: None, no clipping)')
        'clip-mode': 'norm',  #help='Gradient clipping mode. One of ("norm", "value", "agc")')

        # Learning rate schedule parameters
        #'lr-noise': None, #help='learning rate noise on/off epoch percentages')
        'lr-noise-pct': 0.67, #help='learning rate noise limit percent (default: 0.67)')
        'lr-noise-std': 1.0, #help='learning rate noise std-dev (default: 1.0)')
        'lr-cycle-mul': 1.0, #help='learning rate cycle len multiplier (default: 1.0)')
        'lr-cycle-decay': 0.5, #help='amount to decay each learning rate cycle (default: 0.5)')
        'lr-cycle-limit': 1, #help='learning rate cycle limit, cycles enabled if > 1')
        'lr-k-decay' : 1.0, #help='learning rate k-decay for cosine/poly (default: 1.0)')
        'warmup-lr': 0.0001, #help='warmup learning rate (default: 0.0001)')
        'min-lr': 1e-6, #help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        'cooldown-epochs': 10, #help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        'patience-epochs': 10, #help='patience epochs for Plateau LR scheduler (default: 10')
        'decay-rate': 0.1, #help='LR decay rate (default: 0.1)')

        # Augmentation & regularization parameters
        'no-aug': 'False', #help='Disable all training augmentation, override other train aug args')
        'rot': 45,  # metavar='RATIO', help='Random resize aspect ratio (default: 0.75 1.33)')
        'hflip': 0.5, #help='Horizontal flip training aug probability')
        'vflip': 0., #help='Vertical flip training aug probability')
         'contrast': 0.4,
        'brightness': 0.4,
        'saturation': 0,# 'aa': None, # type=str, help='Use AutoAugment policy. "v0" or "original". (default: None)'),
        'aug-repeats': 0, #help='Number of augmentation repetitions (distributed training only) (default: 0)')
        'aug-splits': 0, #help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
        #'bce': True, # todo: fix doppeldeutigkeit
        #'bce-loss': False, #help='Enable BCE loss w/ Mixup/CutMix use.')
        #'bce-target-thresh': None, type=float, help='Threshold for binarizing softened BCE targets (default: None, disabled)')

        'reprob': 0., #metavar='PCT', help='Random erase prob (default: 0.)')
        'remode': 'pixel', #help='Random erase mode (default: "pixel")')
        'recount': 1, # help='Random erase count (default: 1)')
        #'resplit': False, # help='Do not random erase first (clean) augmentation split')

        'mixup': 0.0, # help='mixup alpha, mixup enabled if > 0. (default: 0.)')
        'cutmix': 0.0, # help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
        #'cutmix-minmax': None, # type=float, nargs='+', help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        'mixup-prob': 1.0, #help='Probability of performing mixup or cutmix when either/both is enabled')
        'mixup-switch-prob': 0.5, #help='Probability of switching to cutmix when both mixup and cutmix enabled')
        'mixup-mode': 'batch', #help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
        'mixup-off-epoch': 0, #help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
        'train-interpolation': 'random', #help='Training interpolation (random, bilinear, bicubic default: "random")')

        # Batch norm parameters (only works with gen_efficientnet based models currently)
        #'bn-momentum': None, #type=float, help='BatchNorm momentum override (if not None)')
        #'bn-eps': None, #type=float, default=None, help='BatchNorm epsilon override (if not None)')
        #'sync-bn': False, #help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
        #'dist-bn': 'reduce', #help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
        #'split-bn': False, #help='Enable separate BN layers per augmentation split.')

        # Model Exponential Moving Average
        #'model-ema': False, #help='Enable tracking moving average of model weights')
        #'model-ema-force-cpu': False, #help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
        #'model-ema-decay': 0.9998, #help='decay factor for model weights moving average (default: 0.9998)')

        # Misc
        'seed': 42, #help='random seed (default: 42)')
        'worker-seeding': 'all', #help='worker seed mode (default: all)')
        'log-interval': 50,     #help='how many batches to wait before logging training status')
        'recovery-interval': 0, #help='how many batches to wait before writing recovery checkpoint')
        'checkpoint-hist': 5,  #help='number of checkpoints to keep (default: 10)')
        'workers': 4,           #help='how many training processes to use (default: 4)')
        'save-images': True,   #help='save images of input bathes every log interval for debugging')
        'channels-last': False, #help='Use channels_last memory layout')
        'pin-mem': False,       #help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        'no-prefetcher':  False, # help='disable fast prefetcher')
        'eval-metric': 'auc_roc', #type=str, help='Best metric (default: "auc_roc"')
        'tta': 0,               #type=int, help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
        'use-multi-epochs-loader': False, #help='use the multi-epochs-loader to save time at the beginning of every epoch')
        'torchscript': False,  #help='convert model torchscript for inference')
        'log-wandb': False,     #help='log training and validation metrics to wandb')
    }

    yield {
        **defaults,
    }

def run(**kwargs):

    import torch
    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    label_seed=0
    n_labels = 5000

    datasetname = 'chexpert'
    modelname ='vit_base_patch16_224_miil_in21k'

    # bit_hyperrule
    bit_batchsize = 512
    bit_num_trainsamples = 500*512

    max_batchsize_gpu = 128 # resnet_bitm_50
    grad_accumulator = int(bit_batchsize/max_batchsize_gpu)
    #epochs = bit_num_trainsamples(n_labels)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    import os
    outdir = os.path.splitext(os.path.basename(__file__))[0]


    adapted_args = {
        'output': outdir,
        'experiment' : timestr,
        'data_file_train' : f'../data-local/labels/chexpert/chexpert_{n_labels}_{label_seed:02d}.csv',
        'data_file_val': f'../data-local/chexpert_frap_binary_valid.csv',
        'model': modelname,
        'batch_size' : max_batchsize_gpu,
        'validation-batch-size': 1024,
        'grad-accumulation': grad_accumulator,
        'epochs': 64,
        'sched': 'cosine',  # help = 'LR scheduler (default: "step"')
        'lr': 0.003,  # learning rate
        'decay-epochs': [20, 40, 54],
        'warmup-epochs': 3,  # help='epochs to warmup LR, if scheduler supports')
    }

    args = parse_dict_args(**adapted_args, **kwargs)
    train_timm_multilabel.main(args)
    args = parse_dict_args(**adapted_args, **kwargs)
    train_timm_multilabel.main(args)


if __name__ == "__main__":
    setup_default_logging()
    for run_params in parameters():
        run(**run_params)
