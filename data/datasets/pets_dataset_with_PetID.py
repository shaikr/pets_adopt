import pandas as pd
import numpy as np
import torch
from PIL import Image
from scipy.misc import imread, imsave, imresize
import os

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class PetsDatasetPetID(Dataset):
    def __init__(self, train_imgs_dir, csv_path, label_column, is_train=True):
        """
        Args:
            csv_path (string): path to csv file
            train_imgs_dir (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.imgs_dir = train_imgs_dir
        all_data_info = pd.read_csv(csv_path, dtype={"PetID": str})
        if is_train:
            set_ind = 0
        else:
            set_ind = 1
        set_idxs = all_data_info["set"] == set_ind
        self.data_info = all_data_info.loc[set_idxs]

        # Transforms
        self.to_tensor = transforms.ToTensor()

        self.data_info = self.data_info.loc[(self.data_info['Quantity'] == 1) & (self.data_info['PhotoAmt'] > 0)]
        # Column that contains the image paths
        self.image_arr = np.asarray(self.data_info['PetID']).astype(str)

        # Second column is the labels
        self.label_arr = np.asarray(self.data_info[label_column]).astype('float32')
        # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        try:
            single_image_name = self.image_arr[index] + '-1.jpg'
        except:
            pass
        try:
            img_as_img = Image.open(os.path.join(self.imgs_dir, single_image_name))
        except FileNotFoundError as e:
            single_image_name = '0' + self.image_arr[index] + '-1.jpg'
            img_as_img = Image.open(os.path.join(self.imgs_dir, single_image_name))

        # Open image

        # If not RGB - convert
        if img_as_img.mode != 'RGB':
            img_as_img = img_as_img.convert('RGB')

        img_as_img = img_as_img.resize((224, 224))

        # Make sure correct dimensions (500 x 750)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        pet_id = single_image_name.replace('-1.jpg', '')
        return img_as_tensor, single_image_label, pet_id

    def __len__(self):
        return self.data_len
