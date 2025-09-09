import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import zlib, base64

# import base64


root = "/media/wjy/C/dataset/undersea/dataset/original_data/"
# 定义颜色映射，给每个标签一个独特的颜色 (至少23种颜色)
colors = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 128), (128, 64, 0), (64, 128, 0),
    (0, 64, 128), (128, 0, 64), (0, 128, 64), (64, 128, 128), (128, 128, 64),
    (64, 64, 128), (128, 64, 64), (64, 128, 64)
]
classTitle_list = ['bio', 'rov', 'trash', 'unknown']
img_root = root + "images1/"
mask_root = root + "annotations/"
img_list = os.listdir(img_root)
statistic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
statistic1 = {'bio': 0, 'rov': 0, 'trash': 0, 'unknown': 0}
for i in tqdm(range(len(img_list))):
    file_name = img_list[i]
    img = cv2.imread(img_root + file_name)
    mask_path = mask_root + file_name + ".json"
    # 打开并读取 JSON 文件
    with open(mask_path, 'r') as json_file:
        data = json.load(json_file)
    data_object = data['objects']
    # 叠加mask到原图上
    alpha = 0.4  # 透明度参数
    overlay_image = img.copy()
    mask_file_name_list = os.listdir(mask_root)
    saved_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    exist_label_list = []
    for j in range(len(data_object)):
        # mask_file_name = mask_file_name_list[j]
        mask_data = data_object[j]['bitmap']['data']
        mask_class = data_object[j]['classTitle']
        statistic1[mask_class] += 1
        if mask_class not in exist_label_list:
            exist_label_list.append(mask_class)
        width, height = data_object[j]['bitmap']['origin']
        width, height = int(width), int(height)
        z = zlib.decompress(base64.b64decode(mask_data))
        n = np.fromstring(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3]
        all_mask = np.zeros_like(overlay_image, dtype=np.uint8)
        # try:
        all_mask[height:height + mask.shape[0], width:width + mask.shape[1], 0] = mask
        all_mask[height:height + mask.shape[0], width:width + mask.shape[1], 1] = mask
        all_mask[height:height + mask.shape[0], width:width + mask.shape[1], 2] = mask
        # except:
        #     print("")

        # saved_mask = np.zeros_like(overlay_image, dtype=np.uint8)
        # saved_mask = saved_mask[:, :, 0]
        saved_mask[all_mask == 255] = classTitle_list.index(mask_class) + 1

        color = colors[classTitle_list.index(mask_class)]
        # 创建一个彩色mask
        colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = (all_mask[:, :, c] / 255) * color[c]
        # 将彩色mask以透明的方式叠加到原图上
        overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, alpha, 0)
        # cv2.imwrite("overlay_image.jpg", overlay_image)
        # cv2.imwrite("img.jpg", img)
    statistic[len(exist_label_list)] += 1
    if len(exist_label_list) == 1 and exist_label_list[0] == 'rov':
        # print(exist_label_list)
        cv2.imwrite("/media/wjy/C/dataset/undersea/dataset/original_data/watch/" + file_name, overlay_image)
        save_root = "/media/wjy/C/dataset/undersea/dataset/original_data/masks/"
        cv2.imwrite(save_root + file_name, saved_mask)
        cv2.imwrite("/media/wjy/C/dataset/undersea/dataset/original_data/images/" + file_name, img)
print(statistic)
print(statistic1)
