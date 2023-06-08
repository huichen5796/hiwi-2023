import test_timm_multilabel
from cli import parse_dict_args
from timm.utils import *
import os, torch, yaml

def parameters(): 
    defaults = {
        'dataset': 'chexpert',
        'num-classes': 5,
        'data-dir': '/home/medssl/data/',
        'data_file_val': '../data-local/chexpert_frap_binary_test.csv',

        'crop-pct': 1, # for validation only
        'mean': 0.5,
        'std': 0.5,
        'interpolation': '', #help = 'Image resize interpolation type (overrides model)')

        'validation-batch-size' : 16, #help = 'validation batch size override (default: None)')

    # Model parameters
        'pretrained': 'false',
        #'gp': None #help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
        #'img-size': None #help='Image patch size (default: None => model default)')
        #'input-size': None, #nargs=3, type=int, metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
        'drop': 0.0,  # metavar='PCT', help='Dropout rate (default: 0.)')


        # Misc
        'seed': 42, #help='random seed (default: 42)')
        'worker-seeding': 'all', #help='worker seed mode (default: all)')
        'save-images': True,   #help='save images of input bathes every log interval for debugging')
        'no-prefetcher':  False, # help='disable fast prefetcher')
        'output': '',           #metavar='PATH', help='path to output folder (default: none, current dir)')
    }

    print(os.getcwd())
    directory = './timm2/output/train/'
    #directory = '../../output/train/'
    directory = '../output/train/'
    experiments = sorted(os.listdir(directory))

    for x in experiments:
        run = os.path.join(directory, x)
        yield {
            **defaults,
            'input_path': run,
            'experiment' : x
        }

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def run(**kwargs):


    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    input_path = kwargs['input_path']
    adapted_args = {
        #'labels': 'data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
        'config': os.path.join(input_path, 'args.yaml'),
        'initial-checkpoint': os.path.join(input_path, 'model_best.pth.tar')
    }
    del(kwargs['input_path'])

    args = parse_dict_args(**adapted_args, **kwargs)
    test_timm_multilabel.main(args)


if __name__ == "__main__":
    setup_default_logging()
    for run_params in parameters():
        run(**run_params)
