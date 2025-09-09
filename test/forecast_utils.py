import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, getStat
from torchvision import transforms
from icecream import ic
import wandb

from prompt import Prompt
from medpy import metric
from cal_dice import dice_score, get_res


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


# 将 mask 叠加到原始图像上
# def overlay_masks(image, masks, colors, alpha=0.5):
#     overlay = image.copy()
#     # overlay = overlay * 255
#     for i in range(masks.shape[0]):
#         mask = masks[i]
#         color = colors[i % len(colors)]
#         colored_mask = np.zeros_like(image, dtype=np.uint8)
#         for c in range(3):
#             colored_mask[:, :, c] = mask * color[c]
#         overlay = overlay.astype(colored_mask.dtype)
#         overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
#     return overlay


def overlay_masks(image, masks, colors, alpha=0.5):
    overlay = image.copy()
    # overlay = overlay * 255
    masks = masks[0]
    max_num = masks.max()
    for i in range(1, max_num+1):
        color = colors[i-1]
        mask = (masks == i)
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # 绘制边界
        # mask = mask.astype(np.uint8)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), 1)  # 使用白色绘制边界，线条宽度为2
        overlay = overlay.astype(colored_mask.dtype)
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)  # 使用白色绘制
    return overlay


@torch.no_grad()
def validate(args, model, validloader, multimask_output, img_save_root1, img_save_root2, img_save_root3):
    return_all = args.return_all
    score_dice = []
    model.eval()
    dice_list = None
    dice_dict_list = None
    mean_score_list = []
    for _, sampled_batch in enumerate(validloader):
        print(_)
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        origin_img = cv2.imread(sampled_batch['path'][0])
        origin_img = cv2.resize(origin_img, [256, 256])
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()

        outputs = model(image_batch, multimask_output, args.img_size)

        low_res_logits = outputs['low_res_logits']
        # dice_list = None
        # dice_dict_list = None

        # dice, dice_list = dice_score(low_res_logits, low_res_label_batch, return_all=return_all)

        dice, pred_all, scores_dict = get_res(low_res_logits, low_res_label_batch, return_all=return_all)
        score_dice.append(dice.cpu().numpy())
        # 假设 groundtruth_mask 和 predicted_mask 的形状都是 (n, 64, 64)
        groundtruth_mask = low_res_label_batch.detach().cpu().numpy()
        predicted_mask = pred_all.detach().cpu().numpy()
        # image_batch = image_batch.detach().cpu().numpy()
        # image_batch = image_batch[0].permute(1, 2, 0).detach().cpu().numpy()

        # 调整 groundtruth 和 predicted mask 的大小，使之与原始图像一致
        groundtruth_mask_resized = np.zeros((groundtruth_mask.shape[0], 256, 256), dtype=np.uint8)
        predicted_mask_resized = np.zeros((groundtruth_mask.shape[0], 256, 256), dtype=np.uint8)

        for i in range(groundtruth_mask.shape[0]):
            groundtruth_mask_resized[i] = cv2.resize(groundtruth_mask[i], (256, 256), interpolation=cv2.INTER_NEAREST)
            predicted_mask_resized[i] = cv2.resize(predicted_mask[i], (256, 256), interpolation=cv2.INTER_NEAREST)

        # 定义颜色映射，给每个标签一个独特的颜色
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
                  (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                  (128, 0, 128), (0, 128, 128)]
        # 叠加 groundtruth 和 predicted mask
        groundtruth_overlay = overlay_masks(origin_img, groundtruth_mask_resized, colors, alpha=0.5)
        predicted_overlay = overlay_masks(origin_img, predicted_mask_resized, colors, alpha=0.5)

        mean_score = 0
        num = 0
        for key in scores_dict.keys():
            mean_score += scores_dict[key]
            num += 1
        mean_score = mean_score / num
        mean_score_list.append(mean_score)
        mean_score = round(mean_score, 3)

        # 保存结果
        # cv2.imwrite('original_image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_save_root1+'{}.png'.format(str(_).zfill(4)), groundtruth_overlay)
        cv2.imwrite(img_save_root2+'{}_{}.png'.format(str(_).zfill(4), mean_score), predicted_overlay)
        cv2.imwrite(img_save_root3 + '{}.png'.format(str(_).zfill(4)), origin_img)

        # print("")
    # all_mean_score = np.array(mean_score_list).mean()
    # print("mean_score=", all_mean_score)
    # np.mean(score_dice)
    print("mean_score=", np.mean(score_dice))


def test(args, model, snapshot_path, multimask_output, low_res, test_num=-2000,
            model_num=random.randint(0, 1000000), model_path="",
            img_save_root1="", img_save_root2="", img_save_root3=""):
    if args.dataset == 'kvasir':
        from datasets.dataset_kvasir import Synapse_dataset, RandomGenerator
    elif args.dataset == 'cell':
        from datasets.dataset_cell import Synapse_dataset, RandomGenerator
    elif args.dataset == 'lung':
        from datasets.dataset_lung import Synapse_dataset, RandomGenerator
    elif args.dataset in ['brow', 'eye', 'hair', 'nose', 'mouth', 'celeb']:
        from datasets.dataset_celeb import Synapse_dataset, RandomGenerator
    elif args.dataset in ['car', 'wheel', 'window']:
        from datasets.dataset_car import Synapse_dataset, RandomGenerator
    elif args.dataset == 'teeth':
        from datasets.dataset_teeth import Synapse_dataset, RandomGenerator
    elif args.dataset == 'body':
        from datasets.dataset_body import Synapse_dataset, RandomGenerator
    elif args.dataset == 'data1':
        from datasets.dataset_celebAMMask_HQ import Synapse_dataset, RandomGenerator
    else:
        print("##### Unimplemented dataset #####")
        sys.exit()

    exp_name = f"{args.dataset}-{args.num_data}-{args.exp_type}-lora_mask"
    logger = wandb.init(project='Auto-sam', name=exp_name, resume='allow', anonymous='must', mode=args.wandb_mode)
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base_lr = args.base_lr
    num_classes = args.num_classes

    db_train = Synapse_dataset(
        train_dir=args.root_path, num_data=args.num_data, dataset=args.dataset,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )
    db_test = Synapse_dataset(
        train_dir=args.root_path, num_data=-1, dataset=args.dataset,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )

    num_train = int(len(db_train) * 0.5)
    num_valid = len(db_train) - num_train
    selector = range(len(db_train))
    logging.info("The length of train set is: {}".format(num_train))
    logging.info("The length of valid set is: {}".format(num_valid))

    # print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=32, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:num_train])
    validloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[num_train:])
    selector_test = range(len(db_test))
    testloader = DataLoader(db_test, batch_size=1, num_workers=32, pin_memory=True,
                            worker_init_fn=worker_init_fn, sampler=selector_test[test_num:])

    model.load_state_dict(torch.load(model_path))
    print("load model:", model_path)
    validate(args, model, testloader, multimask_output,
             img_save_root1, img_save_root2, img_save_root3)
    # test_score, test_score_list = validate(args, model, testloader, multimask_output)
    # print('test score : %f' % test_score)
    # if test_score_list is not None:
    #     print("test_score_list:", test_score_list)
    #
    # return "Training Finished!"


if __name__ == '__main__':
    print('test')
