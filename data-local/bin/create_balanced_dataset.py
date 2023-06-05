
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
import torchvision

num_labels = 9*250
dataset_name = 'hist'
datadir = '/home/medssl/data/histology/train'
outdir = 'hist'
noise_ratio = 0 #0.04 # x% label flip

for seed in range(10):

    filename = f'../labels/{outdir}/{dataset_name}_{num_labels}_balanced_{seed:02d}.txt'
    print(filename)

    transform = transforms.RandomHorizontalFlip(p=0)
    dataset = datasets.ImageFolder(datadir, transform)
    num_classes = len(dataset.classes)

    selected_indices = []

    np.random.seed(seed)
    for it, _ in enumerate(dataset.classes):

        class_indices = np.where(np.asarray(dataset.targets)==it)[0]
        bla=5
        selected_indices.extend(np.random.choice(class_indices, replace=False, size=int(num_labels/num_classes+0.5)))


    samples = [list(dataset.samples[ind]) for ind in selected_indices]


    if noise_ratio>0:

        ind_flipped = np.random.choice(range(num_labels), int(num_labels*noise_ratio))

        conf_matrix = np.zeros((num_classes, num_classes))
        for ind in ind_flipped:
            label_idx = samples[ind][1]
            flip = np.random.choice(np.delete(np.arange(num_classes), label_idx))
            conf_matrix[label_idx, flip] += 1
            print(f'old index: {label_idx}, new index: {flip}')
            samples[ind][1] = flip

        print(conf_matrix)



    textfile = open(filename, 'w')

    with textfile:
        for element, i in samples:
            imgfile = os.path.split(element)[1]
            textfile.write(f'{imgfile} {dataset.classes[i]}\n')
    textfile.close()

bla=5