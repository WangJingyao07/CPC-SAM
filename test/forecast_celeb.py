import argparse
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

# from trainer import trainer
from forecast_utils import test
from pathlib import Path
import time


def main(lr1=0.005, lr2=0.005):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root_path', type=str, default=
    # '/media/wjy/C/dataset/CelebAMask-HQ/Images', help='root dir for data')
    # parser.add_argument('--return_all', type=bool,
    #                     default=True,
    #                     help='')
    # parser.add_argument('--output', type=str, default='./saved_model/temp/')
    # parser.add_argument('--dataset', type=str, default='celeb', help='experiment_name')
    # parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
    # parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    # parser.add_argument('--gpu_id', type=str, default='1', help='total gpu')
    # parser.add_argument('--deterministic', type=bool, default=False, help='whether use deterministic training')
    # parser.add_argument('--base_lr', type=float, default=lr1, help='segmentation network learning rate')
    # parser.add_argument('--prompt_base_lr', type=float, default=lr2, help='prompt learning rate')
    # parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    # parser.add_argument('--ckpt', type=str, default='../checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    # parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
    # parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    # parser.add_argument('--warmup', action='store_true',
    #                     help='If activated, warp up the learning from a lower lr to the base_lr')
    # parser.add_argument('--warmup_period', type=int, default=250,
    #                     help='Warp up iterations, only valid whrn warmup is activated')
    # parser.add_argument('--module', type=str, default='sam_lora_mask_decoder')
    # parser.add_argument('--dice_param', type=float, default=0.8)
    #
    # parser.add_argument('--num_data', type=int, default=4, help='batch_size per gpu')
    # parser.add_argument('--exp_type', type=str, default='auto_first')
    #
    # parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    # parser.add_argument('--prompt_weight_decay', type=float, default=0.1, help='weight decay')
    # parser.add_argument('--unrolled', action='store_true', help='')
    # parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/media/wjy/C/dataset/CelebAMask-HQ/Images',
                        help='root dir for data')
    parser.add_argument('--return_all', type=bool,
                        default=True,
                        help='')
    parser.add_argument('--output', type=str, default='./saved_model/temp/')
    parser.add_argument('--dataset', type=str, default='celeb', help='experiment_name')
    parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
    parser.add_argument('--max_epochs', type=int, default=10000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--gpu_id', type=str, default='1', help='total gpu')
    parser.add_argument('--deterministic', type=bool, default=False, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=lr1, help='segmentation network learning rate')
    parser.add_argument('--prompt_base_lr', type=float, default=lr2, help='prompt learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='../checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--warmup', action='store_true',
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--module', type=str, default='sam_lora_mask_decoder')
    parser.add_argument('--dice_param', type=float, default=0.8)

    parser.add_argument('--num_data', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--exp_type', type=str, default='auto_first')

    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--prompt_weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--unrolled', action='store_true', help='')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.exp = args.dataset + str(args.num_data) + '_' + args.exp_type + '_img' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_' + time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1], return_attn=True,
                                                                return_prompt=False,
                                                                return_prompt_weight=False)

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    # print("不可训练数据")
    # # 不可训练数据
    # for n, p in net.named_parameters():
    #     if not p.requires_grad:
    #         print(n, " ", p.shape)
    #
    # print("可训练数据")
    # # 可训练数据
    # for n, p in net.named_parameters():
    #     if p.requires_grad:
    #         print(n, " ", p.shape)

    gt_img_save_root = "../imgs/celeb/4/gt/"
    pred_img_save_root = "../imgs/celeb/4/pred/"
    origin_img_save_root = "../imgs/celeb/4/img/"
    model_path = ("/media/wjy/D/python_code/"
                  "cpc_sam/saved_model/celeb/best_model/355689_4.pth")

    # gt_img_save_root = "../imgs/celeb/8/gt/"
    # pred_img_save_root = "../imgs/celeb/8/pred/"
    # origin_img_save_root = "../imgs/celeb/8/img/"
    # model_path = ("/media/wjy/D/python_code/cpc_sam/"
    #               "saved_model/celeb/best_model/433002_8.pth")

    path = Path(gt_img_save_root)
    path.mkdir(parents=True, exist_ok=True)
    path = Path(pred_img_save_root)
    path.mkdir(parents=True, exist_ok=True)
    path = Path(origin_img_save_root)
    path.mkdir(parents=True, exist_ok=True)

    test(args, net, snapshot_path, multimask_output, low_res, model_path=model_path,
         img_save_root1=gt_img_save_root, img_save_root2=pred_img_save_root, img_save_root3=origin_img_save_root)


if __name__ == "__main__":
    # lr_list = [0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.002,
    #            0.003, 0.004, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012,
    #            0.013, 0.014, 0.015, 0.016, 0.017]
    # # 8样本最佳LR
    lr_1 = 0.005
    lr_2 = 0.005
    # lr_1 = 0.01
    # lr_2 = 0.01
    # for lr_ in lr_list:
    #     print("lr=", lr_)
    #     main(lr_, lr_)
    main(lr_1, lr_2)
