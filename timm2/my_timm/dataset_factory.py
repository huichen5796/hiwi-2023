import os

import numpy as np
import torchvision
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import os
import sys
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pandas as pd

# from timm.data.dataset import IterableImageDataset, ImageDataset
# from data_meanteacher import relabel_dataset, TwoStreamBatchSampler


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)

# copied from torchvision.datasets.folder


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def create_dataset(name, data_dir, classification_type, split='validation', search_split=True, is_training=False):

    # assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    # appends training/validation/test to root path if exists
    if search_split and os.path.isdir(data_dir):
        data_dir = _search_split(data_dir, split)

    if classification_type == 'multiclass':
        # histology: (224, 224), oct: random sizes
        # expects all images of same class in one subdirectory
        dataset = torchvision.datasets.ImageFolder(data_dir)
    elif classification_type == 'multilabel':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return dataset


def create_dataset_from_file(name, data_dir, data_file, classification_type=None, split=None, search_split=False, is_training=False):
    if name == 'chexpert':
        path, dir = os.path.split(data_dir)
        if dir.__contains__('CheXpert'):
            data_dir = path
        return CheXpert(data_dir, data_file)
    else:
        raise NotImplementedError


class DatasetFromFile(VisionDataset):
    """A generic data loader where the samples are read from file: ::

    Args:
        root (string): Root directory path.
        file (string): file wirh image files and labes. format: imgfile label
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            file: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFromFile, self).__init__(root, file, transform=transform,
                                              target_transform=target_transform)

        self.file = file
        self.classes = self._find_classes()
        self.class_to_idx = {self.classes[i]                             : i for i in range(len(self.classes))}
        self.samples = self._make_dataset()
        if len(self.samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(
                    ",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self):
        raise NotImplementedError

    def _make_dataset(self):
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class or array of class targets.
        """
        path, target = self.samples[index]
        imgfile = os.path.join(self.root, path)
        sample = self.loader(imgfile)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class Histology(DatasetFromFile):
    def __init__(
            self,
            root: str,
            file: str,
            loader: Callable[[str], Any],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(DatasetFromFile, self).__init__(root, file, loader, transform=transform,
                                              target_transform=target_transform)

    def _find_classes(self):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(
                dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        raise NotImplementedError


class CheXpert(DatasetFromFile):

    def __init__(
            self,
            root: str,
            file: str,
            loader: Callable[[str], Any] = default_loader,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(CheXpert, self).__init__(root, file, loader, transform=transform,
                                       target_transform=target_transform)

    def _find_classes(self):
        classes = ['Atelectasis', 'Cardiomegaly',
                   'Consolidation', 'Edema', 'Pleural Effusion']
        return classes

    def _make_dataset(self):
        df = pd.read_csv(self.file)
        imgfiles = df['Path'].values
        targets = df[self.classes].values
        targets = np.array([self.flatten(item) for item in targets], dtype=np.float64)
        samples = [s for s in zip(imgfiles, targets)]
        return samples

    def flatten(self, item):
        out = []
        for cla in item:
            print(cla)
            out.extend(eval(cla))
        return out

    def relabel_dataset(self, label_file, no_label_value):
        df = pd.read_csv(label_file)
        imgfiles = df['Path'].values
        targets = df[self.classes].values

        inlier = []
        outlier = []
        for it, (imgfile, target) in enumerate(self.samples):
            if imgfile in imgfiles:
                self.samples[it] = imgfile, target
                inlier.append(it)
            else:
                num_classes = len(self.classes)
                new_target = np.asarray(
                    num_classes*[no_label_value], dtype=np.float64)
                self.samples[it] = imgfile, new_target
                outlier.append(it)

        print(
            f'labeled samples: {len(inlier)} vs. unlabeled samples {len(outlier)}')
        return inlier, outlier
