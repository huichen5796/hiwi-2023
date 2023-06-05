
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import operator

from PIL import ImageStat

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(operator.add, self.h, other.h)))

datadir = '/home/medssl/data/histology/train'

transform = transforms.RandomHorizontalFlip(p=0)
dataset = datasets.ImageFolder(datadir, transform)

imgfileslist = [file for (file, _) in dataset.imgs ]

statistics = None
for imgfile in imgfileslist:
    pil_img = dataset.loader(imgfile)
    if statistics is None:
        statistics = Stats(pil_img)
        bla=5
    else:
        statistics += Stats(pil_img)
        bla=5

print(f'mean:{statistics.mean}, std:{statistics.stddev}')
