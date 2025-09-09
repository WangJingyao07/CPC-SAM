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
import random
import cv2
from torch.utils.data import DataLoader
import time


# class RandomGenerator(object):
#     def __init__(self, output_size, low_res, split=None):
#         self.output_size = output_size
#         self.low_res = low_res
#         self.split = split
#
#     def __call__(self, sample):
#
#         image, label, image_path = sample['image'], sample['label'], sample['path']
#
#         x, y, _ = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
#         # label = np.repeat(np.expand_dims(label, axis=2), 3, 2)
#         label_h, label_w, c = label.shape
#         low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).permute([2,0,1])
#         label = torch.from_numpy(label.astype(np.float32))[:,:,0]
#         low_res_label = torch.from_numpy(low_res_label.astype(np.float32))[:,:,0]
#
#         sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'path': image_path}
#         return sample


class Synapse_dataset(Dataset):
    def __init__(self, train_dir="/data/wjy_data/CelebAMask-HQ/",
                 num_data=0, transform=None, train_flag="train"):
        self.transform = transform  # using transform in torch!
        self.train_file = train_dir
        self.num_data = num_data
        self.train_flag = train_flag
        self.label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                     'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair',
                     'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        self.sample_list = [str(index)+".jpg" for index in range(num_data)]
        s_len = len(self.sample_list)
        # self.sample_list_train = self.sample_list[:int(s_len*0.6)]
        # self.sample_list_valid = self.sample_list[int(s_len * 0.6):int(s_len * 0.8)]
        # self.sample_list_test = self.sample_list[int(s_len * 0.8):int(s_len * 1)]
        self.img_root = train_dir + "CelebA-HQ-img/"
        self.mask_root = train_dir + "CelebAMaskHQ-mask/"

    def __len__(self):
        return len(self.sample_list)
        # if self.train_flag == "train":
        #     return len(self.sample_list_train)
        # elif self.train_flag == "valid":
        #     return len(self.sample_list_valid)
        # else:
        #     return len(self.sample_list_test)

    def __getitem__(self, k):
        # # t_all = time.time()
        # if self.train_flag == "train":
        #     sample_list = self.sample_list_train
        # elif self.train_flag == "valid":
        #     sample_list = self.sample_list_valid
        # elif self.train_flag == "all":
        #     sample_list = self.sample_list
        # else:
        #     sample_list = self.sample_list_test
        sample_list = self.sample_list
        # ['brow', 'eye', 'hair', 'nose', 'mouth']
        # [[204, 0, 204], [76, 153, 0], [204, 0, 0], [51, 51, 255], [0, 255, 255]]
        # t1 = time.time()
        image_path = os.path.join(self.img_root, sample_list[k])
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        label_image_path = os.path.join(self.mask_root, sample_list[k].split(".")[0]+".npy")
        label_img = np.load(label_image_path)
        # label_img = label_img.astype(np.float32)
        image = image / 255.0

        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, axis=2), 3, 2)
        if len(label_img.shape) == 2:
            label_img = np.repeat(np.expand_dims(label_img, axis=2), 3, 2)

        # image = image.transpose(2, 0, 1)
        sample = {'image': image, 'label': label_img, 'path': image_path}
        # t2 = time.time()
        if self.transform is not None:
            sample = self.transform(sample)
        # print("t2=", time.time() - t2)

        # print("t_all = ", time.time() - t_all)
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
        image = torch.from_numpy(image.astype(np.float32)).permute([2, 0, 1])
        label = torch.from_numpy(label.astype(np.float32))[:, :, 0]
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))[:, :, 0]

        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'path': image_path}
        return sample


if __name__ == "__main__":
    train_dataset = Synapse_dataset(train_dir="/data/wjy_data/CelebAMask-HQ/", num_data=30000,
                             transform=None,
                             train_flag="all")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1)
    for batch in train_dataloader:
        images, labels = batch['image'], batch['label'].long()
        print("")

        