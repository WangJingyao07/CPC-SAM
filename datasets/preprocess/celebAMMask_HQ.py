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
from tqdm import tqdm


if __name__ == "__main__":
    root = "/data/wjy_data/CelebAMask-HQ/"
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                  'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair',
                  'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    sample_list = [str(index) + ".jpg" for index in range(30000)]
    img_root = root + "CelebA-HQ-img/"
    mask_root = root + "CelebAMask-HQ-mask-anno/"
    save_root = root + "CelebAMaskHQ-mask/"
    for k in tqdm(range(len(sample_list))):
        img_name = sample_list[k]
        img_id = img_name.split(".")[0]
        folder_num = k // 2000  # 该图片的分割组件存放的目录，一共有15个目录，每个目录存了2000张分割结果（包含一张图片的面部各个组件分开的分割结果）
        # label_img = np.zeros((512, 512, 19))
        label_img = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(mask_root, str(folder_num),
                                    str(k).rjust(5, '0') + '_' + label + '.png')
            if os.path.exists(filename):
                # print(label, idx + 1)
                im = cv2.imread(filename)
                im = im[:, :, 0]  # 取出图像第一个通道的值（分割图像只有一个通道，但是是部分的组件）
                # label_img[:, :, idx+1] = im
                label_img[im != 0] = (idx + 1)  # 将该部分的值赋予一个idx+1的数值，实现分割，后期填充上颜色就变成我们看到的最终分割结果了
        label_img.astype(np.uint8)
        np.save(save_root + img_id + ".npy", label_img)
