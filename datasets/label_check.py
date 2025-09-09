import os
import cv2
import numpy as np


label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
color_list = [[204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

root = "/data/wjy_data/CelebAMask-HQ/"
folder_base = root + 'CelebAMask-HQ-mask-anno/'
folder_save = root + 'CelebAMaskHQ-mask/'
folder_save1 = root + 'CelebAMaskHQ-mask_color/'
img_num = 30000 

if not os.path.exists(folder_save):
    os.mkdir(folder_save)
if not os.path.exists(folder_save1):
    os.mkdir(folder_save1)

new_label_list = []
for k in range(img_num):
    folder_num = k // 2000 
    im_base = np.zeros((512, 512))
    im_base_color = np.zeros((512, 512, 3))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if os.path.exists(filename):
            # print (label, idx+1)
            if label not in new_label_list:
                new_label_list.append(label)
label_list.sort()
new_label_list.sort()
print("")
