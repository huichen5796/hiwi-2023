
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
import torchvision

import pandas

num_labels = 1000
dataset_name = 'chexpert'
datadir = '/home/medssl/data/CheXpert-v1.0-small'
datafile = '/home/medssl/data/CheXpert-v1.0-small/train.csv'
outdir = 'chexpert'
noise_ratio = 0 #0.04 # x% label flip


def split_chexpert(datafile):

    df = pandas.read_csv(datafile)
    patients = [p.split('/')[2] for p in df['Path']]
    df['Patient'] = patients

    pp = np.asarray(list(set(patients)))
    n = len(pp)

    k = round(0.75*n)
    l = round(0.85*n)
    train = pp[:k]
    valid = pp[k:l]
    test = pp[l:]

    df_train = df[df['Patient'].isin(train)]
    df_valid = df[df['Patient'].isin(valid)]
    df_test = df[df['Patient'].isin(test)]

    df_train.to_csv('../chexpert_train.csv', index=False)
    df_valid.to_csv('../chexpert_valid.csv', index=False)
    df_test.to_csv('../chexpert_test.csv', index=False)

def reduced_binary_datasets():
    split='test'
    datafile = f'../chexpert_{split}.csv'

    df = pandas.read_csv(datafile)

    df2 = df[df['Frontal/Lateral']=='Frontal']
    df3 = df2[df2['AP/PA']=='AP']
    #df_gender = df3.groupby('Sex').count()
    df4 = df3.filter(items=['Path', 'Sex', 'Age', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
    class_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    # binarize labels
    df4b = df4

    for c in class_list:
        df4b.loc[df4b[c] == -1, c] = 1
        df4b.loc[df4b[c] != 1, c] = 0

    filename = f'../chexpert_frap_binary_{split}.csv'
    df4b.to_csv(filename, index=False)


def limited_datasets():
    datafile = '../chexpert_train.csv'
    df = pandas.read_csv(datafile)

    df2 = df[df['Frontal/Lateral']=='Frontal']
    df3 = df2[df2['AP/PA']=='AP']
    #df_gender = df3.groupby('Sex').count()
    df4 = df3.filter(items=['Path', 'Sex', 'Age', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
    class_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    # binarize labels
    df4b = df4.copy()

    for c in class_list:
        df4b.loc[df4b[c] == -1, c] = 1
        df4b.loc[df4b[c] != 1, c] = 0

    df_male = df4b[df4b['Sex']=='Male']
    df_female = df4b[df4b['Sex']=='Female']

    for seed in range(10):

        filename = f'../labels/{outdir}/{dataset_name}_{num_labels}_{seed:02d}.txt'
        print(filename)

        df_ms = df_male.sample(n=int(num_labels/2))
        df_fs = df_female.sample(n=int(num_labels/2))

        print(f'{seed: } male mean: {df_ms.mean()}')
        print(f'{seed: } female mean: {df_fs.mean()}')

        df_out = pandas.concat([df_ms, df_fs])

        df_out.to_csv(filename, index=False)


if __name__ == '__main__':
    pass