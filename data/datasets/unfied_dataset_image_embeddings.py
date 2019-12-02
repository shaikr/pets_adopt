from torch.utils.data import Dataset

from data.datasets.pets_dataset_table_step1 import PetsVectorDataset


class UnifiedDatasetImageEmbeddings(Dataset):
    def __init__(self, img_vectors_dir, ds_vectors_dir, csv_path, label_column, is_train=True):
        self.imgs_dataset = PetsVectorDataset(img_vectors_dir, csv_path, label_column, is_train)
        self.ds_vector_dataset = PetsVectorDataset(ds_vectors_dir, csv_path, label_column, is_train)

    def __getitem__(self, index):
        return (self.imgs_dataset.__getitem__(index)[0], self.ds_vector_dataset.__getitem__(index)[0]), \
               self.imgs_dataset.__getitem__(index)[1]

    def __len__(self):
        return len(self.imgs_dataset)
