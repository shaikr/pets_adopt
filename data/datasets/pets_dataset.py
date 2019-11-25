import pandas as pd
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import os

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class PetsDataset(Dataset):
    def __init__(self, imgs_dir, csv_path, label_column, is_train=True):
        """
        Args:
            csv_path (string): path to csv file
            imgs_dir (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        if is_train:
            self.imgs_dir = os.path.join(imgs_dir, 'train_images')
            all_data_info = pd.read_csv(os.path.join(csv_path, 'train.csv'))
        else:
            self.imgs_dir = os.path.join(imgs_dir, 'test_images')
            all_data_info = pd.read_csv(os.path.join(csv_path, 'test.csv'))
        
        # Transforms    
        self.to_tensor = transforms.ToTensor()
        
        self.data_info = all_data_info.loc[all_data_info['Quantity'] == 1]
        # Column that contains the image paths
        self.image_arr = np.asarray(self.data_info['PetID'])
        
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info[label_column])
        # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index] + '-1.jpg'
        # Open image
        img_as_img = Image.open(os.path.join(self.imgs_dir, single_image_name))
        
        # If not RGB - convert
        if img_as_img.mode != 'RGB':
            img_as_img = img_as_img.convert('RGB')
        
        img_as_img = img_as_img.resize((224,224))
        
        # Make sure correct dimensions (500 x 750)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len
