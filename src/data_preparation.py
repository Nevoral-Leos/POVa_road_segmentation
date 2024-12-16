import torch
import cv2
import os
from torch.utils.data import Dataset
from pandas import read_csv

class DeepGlobe(Dataset):
    def __init__(self, dataset_dir, type):
        '''
        Args:
            dataset_dir : The directory where the dataset is located.
            type        : The type of dataset split to load, such as 'train', 'test' or 'valid'. 
                                DO NOT load 'train' and 'test' / 'valid' together, it will most likely crash because maps aren't present for 'test' and  'valid'
        '''
        self.metadata = read_csv(os.path.join(dataset_dir, "metadata.csv"))
        self.type = type
        self.metadata = self.metadata[self.metadata['split'].isin(self.type)]
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        '''
        Returner data format:
                        size                type        range
            img         3 x 1024 x 1024     float32     [0.0, 1.0]
            mask        1024 x 1024         uint8       {0(background), 1(road)}
        '''
        row = self.metadata.iloc[idx]

        sat_image_path = os.path.join(self.dataset_dir, row['sat_image_path'])
        sat_image = cv2.imread(sat_image_path, cv2.IMREAD_COLOR)
        sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB) #convert to RGB because opencv..
        sat_image = torch.tensor(sat_image, dtype=torch.float32).permute(2, 0, 1) / 255.0 #convert to 3x1024x1024 and normalize to [0,1] for convolution layers

        if 'train' in self.type:
            mask_path = os.path.join(self.dataset_dir, row['mask_path'])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = torch.tensor(mask > 127, dtype=torch.uint8) #road 1 background 0
            return sat_image, mask
        return sat_image, None
    