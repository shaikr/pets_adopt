from torch.utils.data import Dataset

from data.datasets.pets_dataset import PetsDataset
from data.datasets.pets_dataset_table_step1 import PetsVectorDataset


class UnifiedDataset(Dataset):
    def __init__(self, train_imgs_dir, vectors_dir, csv_path, label_column, is_train=True):
        self.imgs_dataset = PetsDataset(train_imgs_dir, csv_path, label_column, is_train)
        self.ds_vector_dataset = PetsVectorDataset(vectors_dir, csv_path, label_column, is_train)

    def __getitem__(self, index):
        return (self.imgs_dataset.__getitem__(index)[0], self.ds_vector_dataset.__getitem__(index)[0]), \
               self.imgs_dataset.__getitem__(index)[1]

    def __len__(self):
        return len(self.imgs_dataset)
