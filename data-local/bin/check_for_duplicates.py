import glob
from torchvision import datasets, transforms
import os

traindir = '/home/medssl/data/OCT2017/train'
testdir = '/home/medssl/data/OCT2017/test'

transform = transforms.RandomHorizontalFlip(p=0)
traindata = datasets.ImageFolder(traindir, transform)
train_imgs = set([os.path.split(file)[-1] for (file, _) in traindata.imgs ])

testdata = datasets.ImageFolder(testdir, transform)
test_imgs = set([os.path.split(file)[-1] for (file, _) in testdata.imgs ])


duplicates = train_imgs & test_imgs
print(duplicates)
bla=5