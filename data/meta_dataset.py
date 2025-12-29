# external libs
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import os
import random
from os.path import join as ospj
from glob import glob
# torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
# local libs
from itertools import chain, product
import json
from argparse import Namespace
from torchvision.transforms.functional import InterpolationMode
import sys

sys.path.append("../")
from utils.utils import get_norm_values, chunks
from data.coop import *
from data.randaugment import RandomAugment
from data.randaugment import PatchRandomAugment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "StanfordDogs": "a photo of a {}, a type of dog.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')
        return img


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def dataset_transform(phase, norm_family='clip', rand_aug=False, patch_aug_config=None):
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if patch_aug_config and patch_aug_config['patch_aug_enabled']:
            patch_transform = transforms.Compose([
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                PatchRandomAugment(
                    isPIL=True,
                    patch_size=patch_aug_config['patch_size'],
                    stride=patch_aug_config['stride'],
                    augs=patch_aug_config['aug_ops'],
                    N=patch_aug_config['N'],
                    M=patch_aug_config['M']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            return base_transform, patch_transform
        return base_transform, base_transform

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform, transform
    else:
        raise ValueError('Invalid transform')


def filter_data(all_data, pairs_gt, topk=5):
    valid_files = []
    with open('/home/ubuntu/workspace/top' + str(topk) + '.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1], current[2]))
            attr.append(current[1])
            obj.append(current[2])

    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter += 1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))


DATASET_CLASSMAP = {
    'Caltech101': Caltech101,
    'FGVCAircraft': FGVCAircraft,
    'EuroSAT': EuroSAT,
    'ImageNet': ImageNet,
    'StanfordCars': StanfordCars,
    'DescribableTextures': DTD,
    'Food101': Food101,
    'OxfordPets': OxfordPets,
    'OxfordFlowers': OxfordFlowers,
    'SUN397': SUN397,
    'UCF101': UCF101,
    'ImageNetSketch': ImageNetSketch,
    'ImageNetV2': ImageNetV2,
    'ImageNetA': ImageNetA,
    'ImageNetR': ImageNetR,
}


class MetaDataset(Dataset):
    def __init__(
            self,
            phase,
            dataset=None,
            seed=1,
            return_images=False,
            num_shots=16,
            num_template=1,
            rand_aug=False,
            few_shot=False,
            patch_aug_config=None,
            use_imbalance=False,
            imb_ratio=10.0,
    ):
        self.phase = phase
        self.return_images = return_images
        self.patch_aug_config = patch_aug_config

        dataset_args = Namespace(
            SEED=seed,
            NUM_SHOTS=num_shots,
            SUBSAMPLE_CLASSES='new' if phase == 'test' else 'base',
        )

        if use_imbalance:
            dataset_args.IMBALANCE_RATIO = imb_ratio

        if few_shot:
            dataset_args.SUBSAMPLE_CLASSES = 'all'

        self.dataset = DATASET_CLASSMAP[dataset](dataset_args)
        self.template = CUSTOM_TEMPLATES[dataset]

        self.classnames = self.dataset.classnames
        self.idx2label = self.dataset.lab2cname

        self.loader = ImageLoader('')
        self.transform_base, self.transform_aug = dataset_transform(
            self.phase,
            'clip',
            rand_aug=rand_aug,
            patch_aug_config=patch_aug_config)
        self.num_template = num_template

        self.data_dir = self.dataset.dataset_dir

        if phase == 'train':
            self.dataset = self.dataset.train_x
        else:
            self.dataset = self.dataset.test

    def __getitem__(self, index):
        data_sample = self.dataset[index]

        pil_img = self.loader(data_sample.impath)
        img_1 = self.transform_base(pil_img)
        img_2 = self.transform_aug(pil_img)

        data = [img_1, img_2, data_sample.label]

        if self.return_images:
            data.append(data_sample.impath)

        return data

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from argparse import Namespace
    from flags import DATA_FOLDER

    args = Namespace(
        dataset='FGVCAircraft',
        train_only=True,
        num_shots=16,
        seed=1
    )

    dset = MetaDataset(
        phase='train',
        dataset=args.dataset,
        num_shots=args.num_shots,
        seed=args.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    for d in dataloader:
        pass