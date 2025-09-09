import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


root = "/media/wjy/C/dataset/BBBC038v1/stage1_train/"
file_list = os.listdir(root)
# 定义颜色映射，给每个标签一个独特的颜色 (至少23种颜色)
colors = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 128), (128, 64, 0), (64, 128, 0),
    (0, 64, 128), (128, 0, 64), (0, 128, 64), (64, 128, 128), (128, 128, 64),
    (64, 64, 128), (128, 64, 64), (64, 128, 64)
]
save_root = "/media/wjy/C/dataset/BBBC038v1/"
img_save_root = save_root + "Images/"
mask_save_root = save_root + "Masks/"
watch_save_root = save_root + "Watch/"
img_id_list = []
for i in tqdm(range(len(file_list))):
    file_name = file_list[i]
    root1 = root + file_name + "/"
    img_root = root1 + "images/"
    mask_root = root1 + "masks/"
    img_name = os.listdir(img_root)[0]
    img = cv2.imread(img_root + img_name)
    img_id = img_name.split(".")[0]
    if img_id not in img_id_list:
        img_id_list.append(img_id)
    else:
        print(img_id, "重复拉！！！！！！！！！！1")
    shutil.copy(img_root + img_name, img_save_root + img_name)
    # 叠加mask到原图上
    alpha = 0.4  # 透明度参数
    overlay_image = img.copy()
    mask_file_name_list = os.listdir(mask_root)
    colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    for j in range(len(mask_file_name_list)):
        mask_file_name = mask_file_name_list[j]
        mask = cv2.imread(mask_root + mask_file_name)
        color = colors[j % len(colors)]
        # 创建一个彩色mask
        # colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
        colored_mask[mask == 255] = 255
    #     for c in range(3):
    #         colored_mask[:, :, c] = (mask[:, :, c] / 255) * color[c]
    #     # 将彩色mask以透明的方式叠加到原图上
    #     overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, alpha, 0)
    # cv2.imwrite("overlay_image.jpg", overlay_image)
    # cv2.imwrite("../img.jpg", img)
    # cv2.imwrite("mask.jpg", mask)
    np.save(mask_save_root + img_id + ".png", colored_mask)
    cv2.imwrite(watch_save_root+ img_id + ".jpg", colored_mask)
    # print("")

