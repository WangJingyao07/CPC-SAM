import os
import cv2
import numpy as np
import csv


root = "/media/wjy/C/dataset/lung/segmentation02/segmentation/"
file_list = os.listdir(root)
# 定义颜色映射，给每个标签一个独特的颜色 (至少23种颜色)
colors = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 128), (128, 64, 0), (64, 128, 0),
    (0, 64, 128), (128, 0, 64), (0, 128, 64), (64, 128, 128), (128, 128, 64),
    (64, 64, 128), (128, 64, 64), (64, 128, 64)
]
img_root = root + "Images/"
img_name_list = os.listdir(img_root)
mask_root = root + "Masks/"

for img_name in img_name_list:
    # img_name = img_name_list[i]
    img = cv2.imread(img_root + img_name)
    mask = cv2.imread(mask_root + img_name.split(".")[0] + "_label.png")
    # 叠加mask到原图上
    alpha = 0.4  # 透明度参数
    overlay_image = img.copy()
    colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    cv2.rectangle(colored_mask, img_dict[img_name][0], img_dict[img_name][1], (0, 0, 255), -1)
    saved_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    cv2.rectangle(saved_mask, img_dict[img_name][0], img_dict[img_name][1], (255, 255, 255), -1)
    # mask_file_name_list = os.listdir(mask_root)
    # for j in range(len(mask_file_name_list)):
    #     mask_file_name = mask_file_name_list[j]
    #     mask = cv2.imread(mask_root + mask_file_name)
    #     color = colors[j % len(colors)]
    #     # 创建一个彩色mask
    #     colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    #     for c in range(3):
    #         colored_mask[:, :, c] = (mask[:, :, c] / 255) * color[c]
    #     # 将彩色mask以透明的方式叠加到原图上
    # colored_mask[colored_mask == 255] = colors[0]
    overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, alpha, 0)
    save_root1 = "/media/wjy/C/dataset/X-ray/archive/images/images/"
    save_root2 = "/media/wjy/C/dataset/X-ray/archive/images/masks/"
    save_root3 = "/media/wjy/C/dataset/X-ray/archive/images/watch/"
    cv2.imwrite(save_root1 + img_name, img)
    cv2.imwrite(save_root2 + img_name, saved_mask)
    cv2.imwrite(save_root3 + img_name, overlay_image)
    # cv2.imwrite("overlay_image.jpg", img)
    # cv2.imwrite("img.jpg", img)
    # # cv2.imwrite("mask.jpg", mask)
    # print("")

