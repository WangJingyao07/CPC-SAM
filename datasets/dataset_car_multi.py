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
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2


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
        
        # self.anno = ['BG', 'car', 'wheel', 'window']
        # try:
        #     self.seg_index = self.anno.index(dataset)
        # except:
        #     self.seg_index = None

        if self.num_data > 0:
            sample_list = os.listdir(self.train_file)
            # sample_list.sort()
            self.sample_list = sample_list[:self.num_data]
            print(self.sample_list)
        else:
            self.sample_list = os.listdir(self.train_file)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # ['BG', 'car', 'wheel', 'window']
        # [[0, 0, 0], [204, 0, 204], [76, 153, 0], [51, 51, 255]]
        image_path = os.path.join(self.train_file, self.sample_list[idx])
        image = cv2.imread(image_path).astype(np.float32)
        label = cv2.imread(image_path.replace('/images', '/masks'))
        # label[label == 3] = 0
        # label[label == 4] = 3

        # # label[label == 4] = 0
        # label[label == 1] = 1
        label[label != 0] = 1
        # label[label == 3] = 0
        # label[label == 2] = 1
        image, label = image / 255.0, np.uint8(label)
        # if self.seg_index:
        #     image, label = np.array(image) / 255.0, np.uint8(np.array(label) == self.seg_index)
        # else:
        #     image, label = np.array(image) / 255.0, np.uint8(np.array(label))
        
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, axis=2), 3, 2)
        if len(label.shape) == 2:
            label = np.repeat(np.expand_dims(label, axis=2), 3, 2)
            
        # annot = self.load_annotations(idx)
        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3
        sample = {'image': image, 'label': label, 'path': image_path}
        if self.transform:
            sample = self.transform(sample)
        
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
    root_path = "/media/wjy/C/dataset/car-segmentation/images/"
    db_train = Synapse_dataset(
        train_dir=root_path, num_data=0, dataset="cell",
        transform=transforms.Compose([RandomGenerator(
            output_size=[256, 256], low_res=[64, 64])
        ]),
    )
    selector = range(len(db_train))
    def worker_init_fn(worker_id):
        random.seed(42)
    trainloader = DataLoader(db_train, batch_size=2, num_workers=32, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:2])
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        print("")
        