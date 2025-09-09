import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from PIL import Image
import sys
import csv
import time
from torch.utils.data import DataLoader


class RandomGenerator(object):
    def __init__(self, output_size, low_res, split=None):
        self.output_size = output_size
        self.low_res = low_res
        self.split = split

    def __call__(self, sample):
        
        image, label, image_path = sample['image'], sample['label'], sample['path']
        
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        # label = np.repeat(np.expand_dims(label, axis=2), 3, 2)
        label_h, label_w, c = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1), order=0)
        image = torch.from_numpy(image.astype(np.float32)).permute([2,0,1])
        label = torch.from_numpy(label.astype(np.float32))[:,:,0]
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))[:,:,0]
        
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'path': image_path}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, train_dir, num_data=0, transform=None, dataset='BG'):
        self.transform = transform  # using transform in torch!
        self.train_file = train_dir
        self.num_data = num_data
        
        self.anno = ['BG', 'brow', 'eye', 'hair', 'nose', 'mouth']
        try:
            self.seg_index = self.anno.index(dataset)
        except:
            self.seg_index = None

        file_list = os.listdir(self.train_file)
        # file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        # train_list = file_list[:-2000]
        # test_list = file_list[-2000:]
        # random.shuffle(train_list)
        # file_list = train_list + test_list
        if self.num_data > 0:
            self.sample_list = file_list[:self.num_data]
            # if self.num_data == 4:
            #     self.sample_list = ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
            print(self.sample_list)
        else:
            self.sample_list = file_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # ['brow', 'eye', 'hair', 'nose', 'mouth']
        # [[204, 0, 204], [76, 153, 0], [204, 0, 0], [51, 51, 255], [0, 255, 255]]
        image_path = os.path.join(self.train_file, self.sample_list[idx])
        image = Image.open(image_path)
        # label = Image.open(image_path.replace('/Images', '/Masks').replace('.jpg', '.png'))
        label = np.load(image_path.replace('/Images', '/Masks').replace('.jpg', '.npy'))
        
        if self.seg_index:
            image, label = np.array(image) / 255.0, np.uint8(np.array(label) == self.seg_index)
        else:
            image, label = np.array(image) / 255.0, np.uint8(np.array(label))
        
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, axis=2), 3, 2)
        if len(label.shape) == 2:
            label = np.repeat(np.expand_dims(label, axis=2), 3, 2)
            
        # annot = self.load_annotations(idx)
        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3
        sample = {'image': image, 'label': label, 'path': image_path}
        # t2 = time.time()
        if self.transform:
            sample = self.transform(sample)
        # print("t2=", time.time() - t2)
        
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    
    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)))


if __name__ == "__main__":
    train_dataset = Synapse_dataset(train_dir="/media/wjy/C/dataset/CelebAMask-HQ/Images/",
                                    num_data=4, transform=None, dataset='celeb')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1)
    for batch in train_dataloader:
        images, labels = batch['image'], batch['label'].long()
        print("")
        