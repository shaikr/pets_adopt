# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch.utils import data

from data.datasets.pets_dataset_with_PetID import PetsDatasetPetID
from data.datasets.unified_dataset import UnifiedDataset
from .datasets.mnist import MNIST
from .datasets.im_tmdb import CustomDatasetFromImages
from .datasets.pets_dataset import PetsDataset
from .transforms import build_transforms


def build_dataset(transforms, is_train=True):
    # TODO - currently, no transforms
    # datasets = CustomDatasetFromImages(imgs_dir=r'/data/home/Shai/tmdb/input/posters',
    #                                   csv_path=r"/data/home/Shai/tmdb/input/movies_metadata_with_length_3.csv",
    #                                   label_column=r"vote_average", is_train=is_train)  # "revenue"
    # datasets = PetsDataset(train_imgs_dir=r"/data/home/Shai/petfinder_data/train_images",
    #                        csv_path=r"../data/train.csv",
    #                        label_column=r"AdoptionSpeed", is_train=is_train)
    # datasets = UnifiedDataset(train_imgs_dir=r"/media/ron/Data/google_time/petfinder/train_images",
    #                           vectors_dir="../data/cat_embedded_vectors",
    #                           csv_path=r"../data/train.csv",
    #                           label_column=r"BinaryLabel", is_train=is_train)
    datasets = PetsDatasetPetID(train_imgs_dir=r"/media/ron/Data/google_time/petfinder/train_images",
                                csv_path=r"../data/train.csv",
                                label_column=r"BinaryLabel", is_train=is_train)
    # datasets = MNIST(root='./', train=is_train, transform=transforms, download=True)
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
