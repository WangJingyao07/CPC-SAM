import os
import cv2
import numpy as np
from tqdm import tqdm

# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
#               'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
#               'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
label_list = [['l_brow', 'r_brow'], ['l_eye', 'r_eye'], 'hair', 'nose', 'mouth']
# color_list = [[204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
#               [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
#               [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
color_list = [[204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204]]

# 输入数据集的名称一定要检查一下，这个名称错了不会报错，但是输出的mask里面全部都是空值
root = "/media/wjy/HIKSEMI/dataset/CelebAMask-HQ/"
save_root = "/media/wjy/C/dataset/CelebAMask-HQ/"

folder_base0 = root + 'CelebA-HQ-img/'
folder_base = root + 'CelebAMask-HQ-mask-anno/'
# folder_save = root + 'CelebAMaskHQ-mask/'
folder_save1 = save_root + 'Images/'
folder_save2 = save_root + 'Masks/'
folder_save3 = save_root + 'watch/'
img_num = 30000  # 数据集一共有30000张图片

for k in tqdm(range(img_num)):
    img_name = str(k) + ".jpg"
    img_ori = cv2.imread(folder_base0 + img_name)
    img_ori_resize = cv2.resize(img_ori, [256, 256])
    cv2.imwrite(folder_save1 + img_name, img_ori_resize)
    folder_num = k // 2000  # 该图片的分割组件存放的目录，一共有15个目录，每个目录存了2000张分割结果（包含一张图片的面部各个组件分开的分割结果）
    im_base = np.zeros((512, 512), dtype=np.uint8)
    im_base_color = np.zeros((512, 512, 3), dtype=np.uint8)
    for idx, label in enumerate(label_list):
        if isinstance(label, list):
            for sub_label in label:
                filename = os.path.join(folder_base, str(folder_num),
                                        str(k).rjust(5, '0') +
                                        '_' + sub_label + '.png')
                if os.path.exists(filename):
                    # print(label, idx + 1)
                    im = cv2.imread(filename)
                    im = im[:, :, 0]  # 取出图像第一个通道的值（分割图像只有一个通道，但是是部分的组件）
                    im_base[im != 0] = (idx + 1)  # 将该部分的值赋予一个idx+1的数值，实现分割，后期填充上颜色就变成我们看到的最终分割结果了
                    im_base_color[im != 0] = color_list[idx]
        else:
            filename = os.path.join(folder_base, str(folder_num),
                                    str(k).rjust(5, '0') + '_' + label + '.png')
            if os.path.exists(filename):
                # print(label, idx + 1)
                im = cv2.imread(filename)
                im = im[:, :, 0]  # 取出图像第一个通道的值（分割图像只有一个通道，但是是部分的组件）
                im_base[im != 0] = (idx + 1)  # 将该部分的值赋予一个idx+1的数值，实现分割，后期填充上颜色就变成我们看到的最终分割结果了
                im_base_color[im != 0] = color_list[idx]
    im_base = cv2.resize(im_base, [256, 256], interpolation=cv2.INTER_NEAREST)
    im_base_color = cv2.resize(im_base_color, [256, 256], interpolation=cv2.INTER_NEAREST)
    filename_save2 = os.path.join(folder_save2, str(k) + '.png')
    filename_save3 = os.path.join(folder_save3, str(k) + '.png')
    np.save(folder_save2 + str(k) + ".npy", im_base)
    cv2.imwrite(folder_save3+img_name, im_base_color)
    # print(filename_save)
    # cv2.imwrite(filename_save, im_base)  # 保存图片
    # cv2.imwrite(filename_save1, im_base_color)  # 保存图片
